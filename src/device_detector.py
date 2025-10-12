import re
from typing import Tuple, Optional

try:
    import sounddevice as sd
except (ImportError, OSError) as exc:
    sd = None  # type: ignore[assignment]
    _SD_IMPORT_ERROR = exc
else:
    _SD_IMPORT_ERROR = None


class DeviceDetector:
    """Detect audio devices and their properties for DAF latency adjustment"""
    
    # Bluetooth device name patterns
    BLUETOOTH_PATTERNS = [
        r'airpods',
        r'airpod',
        r'bluetooth',
        r'bt-',
        r'wireless',
        r'headset',
        r'earbuds',
        r'earbud',
        r'beats',
        r'sony wh-',
        r'bose',
        r'samsung galaxy buds',
        r'jabra',
        r'plantronics',
        r'logitech',
        r'jbl',
        r'skullcandy',
        r'anker',
        r'soundcore',
        r'pixel buds',
        r'galaxy buds',
        r'buds live',
        r'buds pro',
        r'buds plus',
        r'buds 2',
        r'buds 2 pro',
        r'buds fe',
        r'buds live',
        r'buds pro 2',
        r'buds pro 3',
        r'buds 3',
        r'buds 3 pro',
        r'buds 3 live',
        r'buds 3 fe',
        r'buds 4',
        r'buds 4 pro',
        r'buds 4 live',
        r'buds 4 fe',
        r'buds 5',
        r'buds 5 pro',
        r'buds 5 live',
        r'buds 5 fe',
        r'buds 6',
        r'buds 6 pro',
        r'buds 6 live',
        r'buds 6 fe',
        r'buds 7',
        r'buds 7 pro',
        r'buds 7 live',
        r'buds 7 fe',
        r'buds 8',
        r'buds 8 pro',
        r'buds 8 live',
        r'buds 8 fe',
        r'buds 9',
        r'buds 9 pro',
        r'buds 9 live',
        r'buds 9 fe',
        r'buds 10',
        r'buds 10 pro',
        r'buds 10 live',
        r'buds 10 fe',
    ]
    
    # Typical Bluetooth latency ranges (in milliseconds)
    BLUETOOTH_LATENCY_RANGES = {
        'low_latency': (50, 100),    # AptX Low Latency, some AirPods
        'standard': (100, 200),      # Standard Bluetooth, most devices
        'high_latency': (200, 300),  # Older Bluetooth, some gaming headsets
    }
    
    def __init__(self):
        if sd is None:
            raise RuntimeError(
                "sounddevice/PortAudio not available. Install PortAudio (e.g. 'sudo apt install libportaudio2') "
                "or set audio_source: esp32_serial."
            ) from _SD_IMPORT_ERROR
        self.devices = sd.query_devices()
        self.default_input = sd.default.device[0]
        self.default_output = sd.default.device[1]
    
    def is_bluetooth_device(self, device_name: str) -> bool:
        """
        Check if a device name matches Bluetooth patterns
        
        Args:
            device_name: Name of the audio device
            
        Returns:
            True if device appears to be Bluetooth
        """
        device_lower = device_name.lower()
        
        for pattern in self.BLUETOOTH_PATTERNS:
            if re.search(pattern, device_lower):
                return True
        
        return False
    
    def get_device_info(self, device_id: Optional[int] = None) -> dict:
        """
        Get detailed information about a device
        
        Args:
            device_id: Device ID (None for default)
            
        Returns:
            Dictionary with device information
        """
        if device_id is None:
            device_id = self.default_output
        
        if device_id >= len(self.devices):
            return {}
        
        device = self.devices[device_id]
        device_name = device.get('name', 'Unknown')
        
        return {
            'id': device_id,
            'name': device_name,
            'is_bluetooth': self.is_bluetooth_device(device_name),
            'sample_rate': device.get('default_samplerate', 48000),
            'max_inputs': device.get('max_inputs', 0),
            'max_outputs': device.get('max_outputs', 0),
        }
    
    def estimate_bluetooth_latency(self, device_name: str) -> int:
        """
        Estimate Bluetooth latency based on device name
        
        Args:
            device_name: Name of the Bluetooth device
            
        Returns:
            Estimated latency in milliseconds
        """
        device_lower = device_name.lower()
        
        # AirPods Pro and newer models typically have lower latency
        if 'airpods pro' in device_lower or 'airpods max' in device_lower:
            return 80  # ~80ms typical latency
        
        # Regular AirPods
        if 'airpods' in device_lower:
            return 120  # ~120ms typical latency
        
        # Gaming headsets often have lower latency
        if any(word in device_lower for word in ['gaming', 'game', 'pro', 'ultra']):
            return 90  # ~90ms typical latency
        
        # Default Bluetooth latency
        return 150  # ~150ms typical latency
    
    def get_daf_adjustment(self, input_device_id: Optional[int] = None, 
                          output_device_id: Optional[int] = None) -> int:
        """
        Estimate Bluetooth latency to compensate in DAF (input + output)
        
        Args:
            input_device_id: Input device ID (None for default)
            output_device_id: Output device ID (None for default)
            
        Returns:
            Estimated Bluetooth mouth-to-ear latency in milliseconds to subtract from target delay
        """
        input_info = self.get_device_info(input_device_id)
        output_info = self.get_device_info(output_device_id)
        
        total_adjustment = 0
        
        # Check input device
        if input_info.get('is_bluetooth', False):
            input_latency = self.estimate_bluetooth_latency(input_info['name'])
            total_adjustment += input_latency
        
        # Check output device
        if output_info.get('is_bluetooth', False):
            output_latency = self.estimate_bluetooth_latency(output_info['name'])
            total_adjustment += output_latency
        
        return total_adjustment
    
    def get_current_devices_info(self) -> dict:
        """
        Get information about current default devices
        
        Returns:
            Dictionary with current device information
        """
        input_info = self.get_device_info(self.default_input)
        output_info = self.get_device_info(self.default_output)
        
        return {
            'input': input_info,
            'output': output_info,
            'daf_adjustment': self.get_daf_adjustment(self.default_input, self.default_output)
        }
    
    def list_devices(self) -> list:
        """
        List all available devices with Bluetooth detection
        
        Returns:
            List of device dictionaries
        """
        device_list = []
        
        for i, device in enumerate(self.devices):
            device_name = device.get('name', 'Unknown')
            is_bluetooth = self.is_bluetooth_device(device_name)
            
            device_info = {
                'id': i,
                'name': device_name,
                'is_bluetooth': is_bluetooth,
                'sample_rate': device.get('default_samplerate', 48000),
                'max_inputs': device.get('max_inputs', 0),
                'max_outputs': device.get('max_outputs', 0),
            }
            
            if is_bluetooth:
                device_info['estimated_latency'] = self.estimate_bluetooth_latency(device_name)
            
            device_list.append(device_info)
        
        return device_list
