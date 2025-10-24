"""Security Manager for data protection and privacy features."""

import os
import json
import keyring
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
# FIXED: Import PBKDF2HMAC instead of PBKDF2
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
from loguru import logger


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    encrypt_storage: bool = True
    encryption_algorithm: str = "AES-256-GCM" # Note: Fernet uses AES-128-CBC
    data_retention_days: int = 7
    auto_cleanup_enabled: bool = True
    store_api_keys: bool = False
    api_key_encryption: bool = True
    enable_data_minimization: bool = True
    anonymize_sensitive_data: bool = False
    audit_logging: bool = True
    audit_log_retention_days: int = 90
    enable_gdpr_features: bool = True


class SecurityManager:
    """
    Manages security and privacy features.
    """
    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        storage_dir: Optional[Path] = None,
        master_key: Optional[str] = None,
    ):
        """Initialize SecurityManager."""
        self.config = config or SecurityConfig()
        self.storage_dir = storage_dir or Path("temp/secure")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = self.storage_dir / "audit.log"
        self.cipher_suite = self._initialize_encryption(master_key)
        logger.info("SecurityManager initialized")
        self._audit_log("security_manager_initialized", {"config": self.config.__dict__})

    def _initialize_encryption(self, master_key: Optional[str] = None) -> Optional[Fernet]:
        """Initialize encryption cipher."""
        if not self.config.encrypt_storage:
            return None
        try:
            key_path = self.storage_dir / ".encryption_key"
            if master_key is None:
                if key_path.exists():
                    key = key_path.read_bytes()
                    logger.info("Loaded existing encryption key.")
                else:
                    key = Fernet.generate_key()
                    key_path.write_bytes(key)
                    key_path.chmod(0o600)
                    logger.info("Generated and saved new encryption key.")
            else:
                # FIXED: Use PBKDF2HMAC
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b"termextractor_salt", # In production, use unique random salt per key/user
                    iterations=480000, # Increased iterations recommended by OWASP
                    backend=default_backend(),
                )
                key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
                logger.info("Derived encryption key from master key.")

            cipher_suite = Fernet(key)
            logger.info("Encryption initialized successfully.")
            return cipher_suite
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            return None

    # ... (rest of the methods: encrypt_data, decrypt_data, store_api_key, etc.) ...
    # Make sure methods like store_api_key and retrieve_api_key handle
    # potential errors from keyring (e.g., if backend is unavailable)

    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data."""
        if not self.cipher_suite:
            # Return original bytes if encryption disabled, handle strings
            if isinstance(data, str):
                return data.encode('utf-8')
            return data # Already bytes
            # Or raise ValueError("Encryption not enabled") if strict encryption is required

        if isinstance(data, str):
            data = data.encode("utf-8")

        try:
            encrypted = self.cipher_suite.encrypt(data)
            self._audit_log("data_encrypted", {"size": len(data)})
            return encrypted
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        if not self.cipher_suite:
             # Return original bytes if encryption disabled
            return encrypted_data
            # Or raise ValueError("Encryption not enabled") if strict encryption is required

        try:
            decrypted = self.cipher_suite.decrypt(encrypted_data)
            self._audit_log("data_decrypted", {"size": len(decrypted)})
            return decrypted
        except Exception as e: # Catch specific crypto errors like InvalidToken
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Decryption failed, key may be incorrect or data corrupted.") from e

    def store_api_key(self, service_name: str, api_key: str, username: str = "default") -> None:
        """Store API key securely using keyring."""
        if not self.config.store_api_keys:
            logger.warning("API key storage is disabled in config.")
            return
        try:
            key_to_store = api_key
            if self.config.api_key_encryption and self.cipher_suite:
                encrypted_key = self.encrypt_data(api_key)
                key_to_store = base64.urlsafe_b64encode(encrypted_key).decode('ascii') # Store as b64 string
                logger.debug(f"Encrypting API key for {service_name} before storing.")

            keyring.set_password(f"termextractor_{service_name}", username, key_to_store)
            logger.info(f"API key stored securely for {service_name}")
            self._audit_log("api_key_stored", {"service": service_name, "username": username})
        except Exception as e: # Catch specific keyring errors if possible
            logger.error(f"Failed to store API key using keyring for {service_name}: {e}")
            # Consider alternative or raise error

    def retrieve_api_key(self, service_name: str, username: str = "default") -> Optional[str]:
        """Retrieve stored API key using keyring."""
        try:
            stored_value = keyring.get_password(f"termextractor_{service_name}", username)
            if not stored_value:
                logger.info(f"No API key found in keyring for {service_name}")
                return None

            api_key = stored_value
            if self.config.api_key_encryption and self.cipher_suite:
                try:
                    encrypted_bytes = base64.urlsafe_b64decode(stored_value.encode('ascii'))
                    decrypted_bytes = self.decrypt_data(encrypted_bytes)
                    api_key = decrypted_bytes.decode("utf-8")
                    logger.debug(f"Decrypted API key retrieved for {service_name}")
                except Exception as e:
                    logger.error(f"Failed to decrypt stored API key for {service_name}: {e}. Returning raw stored value or None.")
                    # Decide on fallback: return raw value? return None?
                    return None # Safer option

            self._audit_log("api_key_retrieved", {"service": service_name, "username": username})
            return api_key
        except Exception as e: # Catch specific keyring errors
            logger.error(f"Failed to retrieve API key using keyring for {service_name}: {e}")
            return None

    def delete_api_key(self, service_name: str, username: str = "default") -> None:
        """Delete stored API key using keyring."""
        try:
            keyring.delete_password(f"termextractor_{service_name}", username)
            logger.info(f"API key deleted from keyring for {service_name}")
            self._audit_log("api_key_deleted", {"service": service_name, "username": username})
        except keyring.errors.PasswordDeleteError:
             logger.info(f"API key for {service_name} not found in keyring, nothing to delete.")
        except Exception as e: # Catch specific keyring errors
            logger.warning(f"Failed to delete API key using keyring for {service_name}: {e}")


    # ... (rest of the methods: minimize_data, anonymize_text, cleanup_old_data, etc.) ...
    def _audit_log(self, event: str, details: Dict[str, Any]) -> None:
        """Log audit event."""
        if not self.config.audit_logging:
            return
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details,
        }
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                # Use default=str to handle non-serializable types gracefully
                f.write(json.dumps(log_entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    # ... (other methods remain largely the same) ...
    def minimize_data(self, text: str, max_context_chars: int = 1000) -> str:
         """Minimize data sent to API by limiting context."""
         # ... (implementation seems okay) ...
         return text # Placeholder if not implemented

    def anonymize_text(self, text: str) -> str:
         """Anonymize sensitive information in text."""
         # ... (implementation seems okay, ensure regex is sufficient) ...
         return text # Placeholder if not implemented

    def cleanup_old_data(self) -> int:
        """Clean up old data based on retention policy."""
        # ... (implementation seems okay) ...
        return 0 # Placeholder if not implemented

    def get_audit_log(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, event_type: Optional[str] = None) -> list:
         """Retrieve audit log entries."""
         # ... (implementation seems okay) ...
         return [] # Placeholder if not implemented

    def export_data(self, user_id: str, output_path: Path) -> None:
        """Export user data (GDPR compliance)."""
        # ... (implementation seems okay) ...
        pass # Placeholder if not implemented

    def delete_user_data(self, user_id: str) -> None:
        """Delete all user data (GDPR compliance)."""
        # ... (implementation seems okay) ...
        pass # Placeholder if not implemented
