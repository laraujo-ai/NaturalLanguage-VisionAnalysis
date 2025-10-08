set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNXRUNTIME_DIR="${SCRIPT_DIR}/onnxruntime"

ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [ "$ARCH" = "aarch64" ]; then
    echo "Installing ONNX Runtime for Jetson (ARM64)..."

    if [ -f /etc/nv_tegra_release ]; then
        JETPACK_VERSION=$(cat /etc/nv_tegra_release | grep -oP 'R\d+' | head -1)
        echo "Detected JetPack: $JETPACK_VERSION"
    else
        echo "Warning: Could not detect JetPack version, using default"
        JETPACK_VERSION="R35"
    fi

    if [[ "$JETPACK_VERSION" == "R35" ]] || [[ "$JETPACK_VERSION" == "R36" ]]; then
        VERSION="1.16.1"
        DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-linux-aarch64-${VERSION}.tgz"
        ARCHIVE_NAME="onnxruntime-linux-aarch64-${VERSION}.tgz"
    else
        echo "ERROR: Unsupported JetPack version: $JETPACK_VERSION"
        echo "Please visit: https://elinux.org/Jetson_Zoo#ONNX_Runtime"
        exit 1
    fi

elif [ "$ARCH" = "x86_64" ]; then
    echo "Installing ONNX Runtime for x86_64..."
    VERSION="1.16.1"
    DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-linux-x64-gpu-${VERSION}.tgz"
    ARCHIVE_NAME="onnxruntime-linux-x64-gpu-${VERSION}.tgz"
else
    echo "ERROR: Unsupported architecture: $ARCH"
    exit 1
fi

echo "Downloading ONNX Runtime v${VERSION}..."
if [ ! -f "$ARCHIVE_NAME" ]; then
    wget --progress=bar:force:noscroll -O "$ARCHIVE_NAME" "$DOWNLOAD_URL"
else
    echo "Archive already exists, skipping download."
fi

if [ -d "$ONNXRUNTIME_DIR" ]; then
    rm -rf "$ONNXRUNTIME_DIR"
fi
mkdir -p "$ONNXRUNTIME_DIR"

cd "$ONNXRUNTIME_DIR"

tar -xzf "../$ARCHIVE_NAME" --strip-components=1

cd "$SCRIPT_DIR"

if [ -f "$ONNXRUNTIME_DIR/lib/libonnxruntime.so" ]; then
    file "$ONNXRUNTIME_DIR/lib/libonnxruntime.so"
else
    exit 1
fi

if [ -f "$ONNXRUNTIME_DIR/include/onnxruntime_cxx_api.h" ]; then
else
    echo "  Looking for headers in: $ONNXRUNTIME_DIR/include/"
    ls -la "$ONNXRUNTIME_DIR/include/" || true
    exit 1
fi

echo ""
echo "Installation complete!"
echo ""

rm -rf onnxruntime
rm $ARCHIVE_NAME