use candle_core::Device;
use anyhow::{Result, anyhow};

/// Converts a device string to a Candle Device.
///
/// # Supported Device Strings
///
/// - `"cpu"`: Returns the CPU device
/// - `"cuda"`: Returns the default CUDA device (index 0)
/// - `"cuda:N"`: Returns the CUDA device with the specified index
///
/// # Arguments
///
/// * `device_str` - A string specifying the desired device
///
/// # Returns
///
/// A `Result` containing the Candle `Device` or an error if the device is not available
///
/// # Examples
///
/// ```
/// // Get the CPU device
/// let cpu_device = get_device("cpu")?;
///
/// // Get the default CUDA device (index 0)
/// let default_cuda = get_device("cuda")?;
///
/// // Get a specific CUDA device
/// let cuda_1 = get_device("cuda:1")?;
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The CUDA device is not available
/// - An unsupported device type is specified
pub fn get_device(device_str: &str) -> Result<Device> {
    if device_str.starts_with("cuda") {
        let cuda_index = if device_str == "cuda" {
            0 // Default to the first CUDA device
        } else {
            device_str.split(':').nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0) 
        };

        let device = Device::cuda_if_available(cuda_index)?;
        if !device.is_cuda() {
            return Err(anyhow!("CUDA device {} is not available", cuda_index));
        }
        Ok(device)
    } else {
        match device_str {
            "cpu" => Ok(Device::Cpu),
            _ => Err(anyhow!("Unsupported device type: {}", device_str)),
        }
    }
}
