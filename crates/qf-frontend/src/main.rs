// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::generate_context;

fn main() {
    tauri::Builder::default()
        .run(generate_context!()) 
        .expect("error while running tauri application");
}
