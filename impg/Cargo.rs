[package]
name = "ibs"
version = "0.1.0"
edition = "2021"
authors = ["Franco Caramia"]
description = "Wrapper around `impg similarity` to obtain IBS segments"
license = "MIT"

[dependencies]
anyhow = "1.0"
clap = { version = "4.4", features = ["derive"] }

[profile.release]
lto = true
codegen-units = 1
strip = true
