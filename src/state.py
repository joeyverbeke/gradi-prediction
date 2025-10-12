import time


class DAFState:
    """State management for DAF system"""
    
    def __init__(self, hold_ms=600):
        """
        Initialize DAF state
        
        Args:
            hold_ms: Hold duration in milliseconds
        """
        self.daf_active = False
        self.hold_ms = hold_ms
        self.hold_start_time = None
    
    def on(self):
        """Activate DAF and start hold timer"""
        self.daf_active = True
        self.hold_start_time = time.time()
    
    def off(self):
        """Deactivate DAF"""
        self.daf_active = False
        self.hold_start_time = None
    
    def update(self):
        """
        Update state and check if hold timer has expired
        
        Returns:
            True if DAF should remain active, False if it should be turned off
        """
        if not self.daf_active or self.hold_start_time is None:
            return False
        
        # Check if hold timer has expired
        elapsed_ms = (time.time() - self.hold_start_time) * 1000
        if elapsed_ms >= self.hold_ms:
            self.off()
            return False
        
        return True
    
    def is_active(self):
        """Check if DAF is currently active"""
        return self.daf_active
    
    def get_remaining_hold_ms(self):
        """Get remaining hold time in milliseconds"""
        if not self.daf_active or self.hold_start_time is None:
            return 0
        
        elapsed_ms = (time.time() - self.hold_start_time) * 1000
        remaining = max(0, self.hold_ms - elapsed_ms)
        return remaining
