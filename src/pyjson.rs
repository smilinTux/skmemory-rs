//! A writer that reproduces Python's `json.dumps` byte for byte.
//!
//! Python's defaults are the compatibility target: `ensure_ascii=True` (every
//! non-ASCII char becomes `\uXXXX`, astral chars as lowercase surrogate pairs),
//! `": "` between key and value, and either `indent=2` (items joined by `",\n"`)
//! or the compact `", "` form. Floats use Python's `repr` (shortest round-trip,
//! scientific only below 1e-4 or from 1e16, `e+16` / `e-05` exponent style).
//! Object key order is the order held by the `serde_json::Map` (the crate is
//! built with `preserve_order`), which is the order Python's dict would have.

use serde_json::{Number, Value};
use std::fmt::Write;

/// Serialize like `json.dumps(value, indent=2)` (no trailing newline).
pub fn dumps_indent2(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value, Some(2), 0);
    out
}

/// Serialize like `json.dumps(value)` (default separators `", "` and `": "`).
pub fn dumps_compact(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value, None, 0);
    out
}

fn newline_indent(out: &mut String, indent: usize, level: usize) {
    out.push('\n');
    for _ in 0..indent * level {
        out.push(' ');
    }
}

fn write_value(out: &mut String, v: &Value, indent: Option<usize>, level: usize) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => out.push_str(&number_repr(n)),
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                match indent {
                    Some(n) => {
                        if i > 0 {
                            out.push(',');
                        }
                        newline_indent(out, n, level + 1);
                    }
                    None if i > 0 => out.push_str(", "),
                    None => {}
                }
                write_value(out, item, indent, level + 1);
            }
            if let Some(n) = indent {
                newline_indent(out, n, level);
            }
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, item)) in map.iter().enumerate() {
                match indent {
                    Some(n) => {
                        if i > 0 {
                            out.push(',');
                        }
                        newline_indent(out, n, level + 1);
                    }
                    None if i > 0 => out.push_str(", "),
                    None => {}
                }
                write_string(out, k);
                out.push_str(": ");
                write_value(out, item, indent, level + 1);
            }
            if let Some(n) = indent {
                newline_indent(out, n, level);
            }
            out.push('}');
        }
    }
}

/// Python `json` string encoding with `ensure_ascii=True`.
pub fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{:04x}", unit);
                }
            }
        }
    }
    out.push('"');
}

/// Python `json` number output. The crate runs serde_json with
/// `arbitrary_precision`, so a number keeps its source text: an integer
/// literal (any size) is emitted as Python's `int` would be, anything with a
/// fraction or exponent goes through `float` and `repr`.
fn number_repr(n: &Number) -> String {
    let text = n.to_string();
    if text.bytes().any(|b| matches!(b, b'.' | b'e' | b'E')) {
        float_repr(text.parse::<f64>().unwrap_or(f64::NAN))
    } else if text == "-0" {
        "0".into()
    } else {
        text
    }
}

/// Python's `float.__repr__` (what `json.dumps` uses for floats).
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // `{:e}` yields the shortest round-trip digits: "-1.2345e-5".
    let sci = format!("{:e}", x);
    let (mant, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp < 0 {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        } else {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                out.push_str(&digits);
                for _ in 0..(int_len - digits.len()) {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (5.0, "5.0"),
            (0.5, "0.5"),
            (8.5, "8.5"),
            (-0.9, "-0.9"),
            (0.1, "0.1"),
            (0.30000000000000004, "0.30000000000000004"),
            (1e-4, "0.0001"),
            (1.5e-5, "1.5e-05"),
            (1e-5, "1e-05"),
            (123456789012345.0, "123456789012345.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e20, "1.2345e+20"),
            (1e100, "1e+100"),
            (2.5e-300, "2.5e-300"),
            (123.456, "123.456"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(*x), *want, "{x:e}");
        }
    }

    #[test]
    fn integers_keep_arbitrary_precision() {
        let v: Value = serde_json::from_str(
            r#"[123456789012345678901234567890, -0, 1E5, 1e400, 7, -9223372036854775809]"#,
        )
        .unwrap();
        assert_eq!(
            dumps_compact(&v),
            "[123456789012345678901234567890, 0, 100000.0, Infinity, 7, -9223372036854775809]"
        );
    }

    #[test]
    fn strings_match_ensure_ascii() {
        let mut s = String::new();
        write_string(&mut s, "a\"b\\c\n\t\u{1}\u{7f}é😀");
        assert_eq!(s, r#""a\"b\\c\n\t\u0001\u007f\u00e9\ud83d\ude00""#);
    }

    #[test]
    fn indent2_layout() {
        let v: Value = serde_json::from_str(r#"{"a":[],"b":{},"c":[1,{"d":null}]}"#).unwrap();
        assert_eq!(
            dumps_indent2(&v),
            "{\n  \"a\": [],\n  \"b\": {},\n  \"c\": [\n    1,\n    {\n      \"d\": null\n    }\n  ]\n}"
        );
        assert_eq!(
            dumps_compact(&v),
            r#"{"a": [], "b": {}, "c": [1, {"d": null}]}"#
        );
    }
}
