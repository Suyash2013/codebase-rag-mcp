"""
Model Router Component for Langflow
====================================
Smart routing between cloud APIs (Gemini) and local models (Ollama)
based on API key availability and quota.
"""

import os
import httpx
from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.io import SecretStrInput, StrInput, IntInput, MessageTextInput
from lfx.schema.message import Message
from lfx.template.field.base import Output


class ModelRouterComponent(Component):
    display_name = "Model Router"
    description = "Routes to Gemini API if available, falls back to local Ollama models."
    icon = "route"
    name = "ModelRouter"

    inputs = [
        SecretStrInput(
            name="gemini_api_key",
            display_name="Gemini API Key",
            info="Google AI Studio API key. If empty or invalid, uses Ollama fallback.",
            required=False,
        ),
        StrInput(
            name="gemini_model",
            display_name="Gemini Model",
            info="Gemini model to use when API is available",
            value="gemini-1.5-flash",
        ),
        StrInput(
            name="ollama_primary",
            display_name="Ollama Primary Fallback",
            info="Primary Ollama model to use when Gemini is unavailable",
            value="qwen2.5:7b",
        ),
        StrInput(
            name="ollama_secondary",
            display_name="Ollama Secondary Fallback",
            info="Secondary Ollama model (emergency fallback)",
            value="codellama:7b",
        ),
        StrInput(
            name="ollama_base_url",
            display_name="Ollama Base URL",
            info="Ollama API base URL",
            value="http://localhost:11434",
            advanced=True,
        ),
        IntInput(
            name="timeout_seconds",
            display_name="API Check Timeout",
            info="Timeout in seconds for checking API availability",
            value=5,
            advanced=True,
        ),
        MessageTextInput(
            name="prompt",
            display_name="Prompt",
            info="The prompt to send to the selected model",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(display_name="Response", name="response", method="route_and_generate"),
        Output(display_name="Selected Provider", name="provider_info", method="get_provider_info"),
    ]

    def _check_gemini_available(self) -> bool:
        """Check if Gemini API is available and has quota."""
        if not self.gemini_api_key:
            self.log("No Gemini API key provided")
            return False
        
        try:
            # Quick API check using a minimal request
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.gemini_api_key}"
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(url)
                if response.status_code == 200:
                    self.log("✅ Gemini API available")
                    return True
                elif response.status_code == 429:
                    self.log("⚠️ Gemini quota exceeded, using fallback")
                    return False
                else:
                    self.log(f"⚠️ Gemini API error: {response.status_code}")
                    return False
        except Exception as e:
            self.log(f"⚠️ Gemini check failed: {e}")
            return False

    def _check_ollama_model(self, model: str) -> bool:
        """Check if an Ollama model is available."""
        try:
            url = f"{self.ollama_base_url}/api/tags"
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(url)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    available = any(m.get("name", "").startswith(model.split(":")[0]) for m in models)
                    return available
                return False
        except Exception:
            return False

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]

    def _call_ollama(self, model: str, prompt: str) -> str:
        """Call Ollama API."""
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        with httpx.Client(timeout=120) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]

    def route_and_generate(self) -> Message:
        """Route to best available model and generate response."""
        
        # Try Gemini first
        if self._check_gemini_available():
            try:
                self.log(f"Using Gemini: {self.gemini_model}")
                response = self._call_gemini(self.prompt)
                return Message(text=response)
            except Exception as e:
                self.log(f"Gemini failed: {e}, trying Ollama fallback")
        
        # Try primary Ollama fallback
        if self._check_ollama_model(self.ollama_primary):
            try:
                self.log(f"Using Ollama: {self.ollama_primary}")
                response = self._call_ollama(self.ollama_primary, self.prompt)
                return Message(text=response)
            except Exception as e:
                self.log(f"Ollama primary failed: {e}")
        
        # Try secondary Ollama fallback
        if self._check_ollama_model(self.ollama_secondary):
            try:
                self.log(f"Using Ollama: {self.ollama_secondary}")
                response = self._call_ollama(self.ollama_secondary, self.prompt)
                return Message(text=response)
            except Exception as e:
                self.log(f"Ollama secondary failed: {e}")
        
        return Message(text="❌ No models available. Please check your configuration.")

    def get_provider_info(self) -> Message:
        """Return info about which provider would be selected."""
        if self._check_gemini_available():
            return Message(text=f"Provider: Google GenAI | Model: {self.gemini_model}")
        elif self._check_ollama_model(self.ollama_primary):
            return Message(text=f"Provider: Ollama | Model: {self.ollama_primary}")
        elif self._check_ollama_model(self.ollama_secondary):
            return Message(text=f"Provider: Ollama | Model: {self.ollama_secondary}")
        else:
            return Message(text="No providers available")
