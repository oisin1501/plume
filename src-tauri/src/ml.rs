use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn find_python3() -> String {
    let candidates = [
        "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
    ];

    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return path.to_string();
        }
    }

    // Fall back to PATH lookup
    "python3".to_string()
}

pub fn run_ml_command(command: serde_json::Value) -> Result<serde_json::Value, String> {
    let python_script = std::env::current_exe()
        .map_err(|e| format!("Failed to get exe path: {}", e))?
        .parent()
        .ok_or("No parent dir")?
        .join("../../../python/plume_ml.py")
        .canonicalize()
        .or_else(|_| {
            // Try dev path
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../python/plume_ml.py")
                .canonicalize()
                .map_err(|e| format!("Cannot find plume_ml.py: {}", e))
        })?;

    // Try to find python3 — bundled .app doesn't inherit the user's shell PATH
    let python = find_python3();

    let mut child = Command::new(&python)
        .arg(&python_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start Python (tried '{}'): {}", python, e))?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    let stdout = child.stdout.take().ok_or("Failed to open stdout")?;

    let cmd_str = serde_json::to_string(&command)
        .map_err(|e| format!("Serialization error: {}", e))?;

    stdin
        .write_all(format!("{}\n", cmd_str).as_bytes())
        .map_err(|e| format!("Write error: {}", e))?;
    drop(stdin); // Close stdin so Python knows to process and exit

    let reader = BufReader::new(stdout);
    let mut result_line = String::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Read error: {}", e))?;
        if !line.trim().is_empty() {
            result_line = line;
            break;
        }
    }

    let _ = child.wait();

    if result_line.is_empty() {
        // Check stderr for errors
        let stderr = child.stderr.take();
        if let Some(stderr) = stderr {
            let err_reader = BufReader::new(stderr);
            let err_lines: Vec<String> = err_reader.lines().filter_map(|l| l.ok()).collect();
            if !err_lines.is_empty() {
                return Err(err_lines.join("\n"));
            }
        }
        return Err("No response from Python".to_string());
    }

    let result: serde_json::Value = serde_json::from_str(&result_line)
        .map_err(|e| format!("Failed to parse Python response: {} — raw: {}", e, result_line))?;

    if let Some(err) = result.get("error") {
        return Err(err.as_str().unwrap_or("Unknown error").to_string());
    }

    Ok(result)
}
