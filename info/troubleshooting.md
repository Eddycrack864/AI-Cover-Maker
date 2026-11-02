Here are some common errors that could happen during the installation or running process.
To see what AI Cover Maker runs on (Python, GPU, CUDA, FFmpeg), run `info/status-checker.bat` (Windows) or
`info/status-checker.sh` (Linux) and include its output when you ask for help.

## 1. FFmpeg not found

### What's this?

AI Cover Maker needs FFmpeg to read and write audio, and it must be on the PATH.

### How to solve it?

- **Windows**: run `info/ffmpeg-installer.bat`, then close and open again the window of AI Cover Maker (the new PATH
  only applies to new windows). Alternatively: `winget install Gyan.FFmpeg`.
- **Linux**: `sudo apt install ffmpeg` (Debian/Ubuntu), `sudo pacman -S ffmpeg` (Arch) or `sudo dnf install ffmpeg`
  (Fedora).

## 2. The cover runs on the CPU, not on my NVIDIA GPU

### What's this?

PyTorch couldn't use your GPU, so everything runs on the CPU, which is much slower. The status checker says
`no NVIDIA GPU found`.

### How to solve it?

- Update your NVIDIA driver: AI Cover Maker uses CUDA 12.8, which needs a recent driver.
- Only NVIDIA GPUs are supported (GTX 10 series or newer). AMD and Intel GPUs, and Apple Silicon, run on the CPU.
- If you don't have a supported GPU, use the [Colab](https://colab.research.google.com/github/Eddycrack864/AI-Cover-Maker/blob/main/AI_Cover_Maker_Colab.ipynb).

## 3. CUDA out of memory

### What's this?

The GPU doesn't have enough memory for the chosen model.

### How to solve it?

- Close other programs that use the GPU (games, other AI apps) and try again.
- Choose lighter separation models (the **Fast** preset, or MDX-NET / VR Arch models instead of the big Roformer ones).
- On GPUs with 8 GB or less, AI Cover Maker already avoids keeping two models loaded at the same time.

## 4. "Could not tell which stem to keep for the ... step"

### What's this?

The model chosen for that separation step doesn't output what the step needs, e.g. an instrumental model in the
vocals step. This can only happen with **Show all models** checked.

### How to solve it?

Uncheck **Show all models** in the **Separation steps** and choose one of the models listed for that step.

## 5. HF_HUB_DISABLE_SYMLINKS_WARNING

### What's this?

This message is a warning, not an error. On Windows without developer mode, the Hugging Face cache can't use symlinks,
so it uses a bit more disk space.

### How to solve it?

You can ignore it, or follow the steps described [here](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development).

## 6. No module named 'xxxxxx'

### What's this?

The dependencies were not installed correctly, or the scripts can't find them.

### How to solve it?

- Make sure you installed AI Cover Maker in a folder that does not require administrator permissions.
- Make sure the installation path does not have spaces or special characters, e.g. `C:\AI-Cover-Maker`.
- If you downloaded the `.zip` of a release, rename the folder from `AI-Cover-Maker-X.X.X` to just `AI-Cover-Maker`.
- Run the updater to install the missing dependencies, or delete the `env` folder and run the installer again.

## 7. Failed building wheel (Linux)

### What's this?

On Linux, one of the dependencies has no prebuilt package and is compiled during the installation. This error means no
C compiler was found.

### How to solve it?

Install a compiler and run the installer or updater again: `sudo apt install build-essential` (Debian/Ubuntu),
`sudo pacman -S base-devel` (Arch) or `sudo dnf install gcc gcc-c++` (Fedora).

## 8. The precompiled version doesn't start on Linux (GLIBC not found)

### What's this?

The Linux precompiled version needs glibc 2.35 or newer.

### How to solve it?

Use Ubuntu 22.04, Debian 12, Fedora 36 or a newer distribution, or do a normal installation with
`AI-Cover-Maker-installer.sh`.

## 9. The port is already in use

### What's this?

Another program (or another AI Cover Maker window) is already using port 7755.

### How to solve it?

Close the other AI Cover Maker window, or use another port: `run-AI-Cover-Maker.bat --listen-port 8080`.

## 10. A voice model can't be downloaded

### What's this?

The link is not a direct or public link to the model.

### How to solve it?

- Use a Hugging Face link, a public Google Drive link, or a direct link to a `.zip`, `.pth` or `.index` file.
- Or download the files yourself and use **Upload a voice model**, or copy them to `models/rvc/<name>/` and press
  **Refresh voice models**.

## 11. How to disable Discord Rich Presence?

1. Open AI Cover Maker and go to the `Settings` tab.
2. Open the `Discord Rich Presence` section and uncheck `Show AI Cover Maker activity on Discord`.
3. Restart AI Cover Maker.

The setting is saved in `user_data/config.json` (`"discord_presence": false`).

## 12. The Python environment rebuild failed

### What's this?

When a new version of AI Cover Maker needs a newer Python, the updater offers to rebuild the Python environment. If the
rebuild fails (no internet, not enough disk space, AI Cover Maker was open...), your previous environment is restored,
so AI Cover Maker keeps working as before.

### How to solve it?

- Close AI Cover Maker before running the updater.
- Make sure you have about 6 GB of free disk space and a stable internet connection.
- Run the updater again and accept the rebuild.
- If an `env_old` folder is left behind and `env` is missing, rename `env_old` to `env`.

If your error is not described here, please create an [issue](https://github.com/Eddycrack864/AI-Cover-Maker/issues)
with the output of the status checker and the error.
