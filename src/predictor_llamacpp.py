import os
from typing import Optional
from llama_cpp import Llama


class PredictorLlamaCPP:
    """Next-token predictor using llama-cpp-python"""
    
    def __init__(
        self,
        model_path: str = "models/llm/llama-3.2-1b-q4_k_m.gguf",
        context_tokens: int = 96,
        n_gpu_layers: Optional[int] = None,
    ):
        """
        Initialize LLM predictor
        
        Args:
            model_path: Path to GGUF model file
            context_tokens: Maximum context tokens to use
            n_gpu_layers: Number of layers to offload to GPU (-1 for auto, 0 for CPU)
        """
        self.context_tokens = context_tokens
        self.n_gpu_layers = n_gpu_layers
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LLM model not found at {model_path}")
        
        # Load LLM model
        try:
            llama_kwargs = {
                "model_path": model_path,
                "n_ctx": 1024,
                "logits_all": False,
                "verbose": False,
            }
            # Only pass GPU layers when explicitly provided so CPU-only setups keep working.
            if self.n_gpu_layers is not None:
                llama_kwargs["n_gpu_layers"] = self.n_gpu_layers

            self.model = Llama(
                **llama_kwargs
            )
            print(f"Loaded LLM model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM model: {e}")
    
    def predict_horizon(self, rolling_text: str, params: dict) -> str:
        """
        Predict next tokens (horizon)
        
        Args:
            rolling_text: Current text context
            params: Prediction parameters (min_pred_tokens, max_pred_tokens, top_k)
            
        Returns:
            Predicted continuation as lowercase string
        """
        if not rolling_text.strip():
            return ""
        
        # Truncate to last context_tokens by simple tail truncation
        words = rolling_text.split()
        if len(words) > self.context_tokens:
            truncated_text = " ".join(words[-self.context_tokens:])
        else:
            truncated_text = rolling_text
        
        # Prepare prompt
        prompt = truncated_text.strip()
        
        try:
            # Generate prediction
            response = self.model(
                prompt,
                max_tokens=params.get('max_pred_tokens', 5),
                temperature=0.0,
                top_k=params.get('top_k', 20),
                stop=["\n"],
                echo=False
            )
            
            # Extract generated text
            generated_text = response['choices'][0]['text'].strip()
            
            # Return lowercase prediction
            return generated_text.lower()
            
        except Exception as e:
            print(f"LLM prediction error: {e}")
            return ""
    
    def test_prediction(self, text: str) -> str:
        """
        Test prediction with default parameters
        
        Args:
            text: Input text
            
        Returns:
            Predicted continuation
        """
        params = {
            'min_pred_tokens': 3,
            'max_pred_tokens': 5,
            'top_k': 20
        }
        return self.predict_horizon(text, params)
