use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// Locate the bundled `plume_ml` binary.
/// Production (inside a macOS .app): sits next to the main executable.
/// Development: fall back to `python3 python/plume_ml.py`.
enum MlBackend {
    Bundled(std::path::PathBuf),
    PythonScript { python: String, script: std::path::PathBuf },
}

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

fn resolve_backend() -> Result<MlBackend, String> {
    let exe_dir = std::env::current_exe()
        .map_err(|e| format!("Failed to get exe path: {}", e))?
        .parent()
        .ok_or("No parent dir")?
        .to_path_buf();

    // 1. Check for bundled binary next to the app executable (production — onefile)
    let bundled = exe_dir.join("plume_ml");
    if bundled.exists() && bundled.is_file() {
        return Ok(MlBackend::Bundled(bundled));
    }

    // 2. Check Tauri resource dir for onedir bundle (production — macOS .app)
    //    Resources are copied to Contents/Resources/ in the .app bundle
    let resource_candidates = [
        exe_dir.join("../Resources/plume_ml/plume_ml"),
        exe_dir.join("../Resources/plume_ml"),
        exe_dir.join("plume_ml/plume_ml"),
    ];
    for candidate in &resource_candidates {
        if let Ok(canonical) = candidate.canonicalize() {
            if canonical.exists() && canonical.is_file() {
                return Ok(MlBackend::Bundled(canonical));
            }
        }
    }

    // 3. Development fallback: run python3 plume_ml.py
    let script = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../python/plume_ml.py")
        .canonicalize()
        .map_err(|e| format!("Cannot find plume_ml.py: {}", e))?;

    let python = find_python3();
    Ok(MlBackend::PythonScript { python, script })
}

pub fn run_ml_command(command: serde_json::Value) -> Result<serde_json::Value, String> {
    let backend = resolve_backend()?;

    let mut child = match &backend {
        MlBackend::Bundled(path) => Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start bundled plume_ml ('{}'): {}", path.display(), e))?,
        MlBackend::PythonScript { python, script } => Command::new(python)
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start Python (tried '{}'): {}", python, e))?,
    };

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    let stdout = child.stdout.take().ok_or("Failed to open stdout")?;

    let cmd_str = serde_json::to_string(&command)
        .map_err(|e| format!("Serialization error: {}", e))?;

    stdin
        .write_all(format!("{}\n", cmd_str).as_bytes())
        .map_err(|e| format!("Write error: {}", e))?;
    drop(stdin); // Close stdin so the sidecar knows to process and exit

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
        return Err("No response from ML sidecar".to_string());
    }

    let result: serde_json::Value = serde_json::from_str(&result_line)
        .map_err(|e| format!("Failed to parse sidecar response: {} — raw: {}", e, result_line))?;

    if let Some(err) = result.get("error") {
        return Err(err.as_str().unwrap_or("Unknown error").to_string());
    }

    Ok(result)
}
