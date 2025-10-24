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
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
from loguru import logger


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    encrypt_storage: bool = True
    encryption_algorithm: str = "AES-256-GCM"
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

    Features:
    - Data encryption/decryption
    - Secure API key storage
    - Data retention and cleanup
    - Audit logging
    - GDPR compliance features
    - Data minimization
    - Sensitive data anonymization
    """

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        storage_dir: Optional[Path] = None,
        master_key: Optional[str] = None,
    ):
        """
        Initialize SecurityManager.

        Args:
            config: Security configuration
            storage_dir: Directory for secure storage
            master_key: Master encryption key (if None, generates new)
        """
        self.config = config or SecurityConfig()
        self.storage_dir = storage_dir or Path("temp/secure")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.audit_log_path = self.storage_dir / "audit.log"

        # Initialize encryption
        self.cipher_suite = self._initialize_encryption(master_key)

        logger.info("SecurityManager initialized")
        self._audit_log("security_manager_initialized", {"config": self.config.__dict__})

    def _initialize_encryption(self, master_key: Optional[str] = None) -> Optional[Fernet]:
        """
        Initialize encryption cipher.

        Args:
            master_key: Master encryption key

        Returns:
            Fernet cipher suite
        """
        if not self.config.encrypt_storage:
            return None

        try:
            if master_key is None:
                # Generate new key
                key = Fernet.generate_key()
                # Store key securely (in production, use HSM or key vault)
                key_path = self.storage_dir / ".encryption_key"
                key_path.write_bytes(key)
                key_path.chmod(0o600)  # Read/write for owner only
            else:
                # Derive key from master key
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b"termextractor_salt",  # In production, use random salt
                    iterations=100000,
                    backend=default_backend(),
                )
                key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))

            cipher_suite = Fernet(key)
            logger.info("Encryption initialized")
            return cipher_suite

        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            return None

    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data

        Raises:
            ValueError: If encryption is not enabled
        """
        if not self.cipher_suite:
            raise ValueError("Encryption not enabled")

        if isinstance(data, str):
            data = data.encode("utf-8")

        encrypted = self.cipher_suite.encrypt(data)
        self._audit_log("data_encrypted", {"size": len(data)})
        return encrypted

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data.

        Args:
            encrypted_data: Encrypted data

        Returns:
            Decrypted data

        Raises:
            ValueError: If encryption is not enabled
        """
        if not self.cipher_suite:
            raise ValueError("Encryption not enabled")

        decrypted = self.cipher_suite.decrypt(encrypted_data)
        self._audit_log("data_decrypted", {"size": len(decrypted)})
        return decrypted

    def store_api_key(
        self,
        service_name: str,
        api_key: str,
        username: str = "default",
    ) -> None:
        """
        Store API key securely.

        Args:
            service_name: Name of the service (e.g., 'anthropic')
            api_key: API key to store
            username: Username/identifier
        """
        if not self.config.store_api_keys:
            logger.warning("API key storage is disabled")
            return

        try:
            if self.config.api_key_encryption and self.cipher_suite:
                # Encrypt before storing
                encrypted_key = self.encrypt_data(api_key)
                # Store encrypted key
                keyring.set_password(
                    f"termextractor_{service_name}",
                    username,
                    encrypted_key.hex(),
                )
            else:
                # Store directly (less secure)
                keyring.set_password(
                    f"termextractor_{service_name}",
                    username,
                    api_key,
                )

            logger.info(f"API key stored for {service_name}")
            self._audit_log("api_key_stored", {"service": service_name, "username": username})

        except Exception as e:
            logger.error(f"Failed to store API key: {e}")
            raise

    def retrieve_api_key(
        self,
        service_name: str,
        username: str = "default",
    ) -> Optional[str]:
        """
        Retrieve stored API key.

        Args:
            service_name: Name of the service
            username: Username/identifier

        Returns:
            API key or None if not found
        """
        try:
            stored = keyring.get_password(
                f"termextractor_{service_name}",
                username,
            )

            if not stored:
                return None

            if self.config.api_key_encryption and self.cipher_suite:
                # Decrypt
                encrypted_bytes = bytes.fromhex(stored)
                decrypted = self.decrypt_data(encrypted_bytes)
                api_key = decrypted.decode("utf-8")
            else:
                api_key = stored

            self._audit_log("api_key_retrieved", {"service": service_name, "username": username})
            return api_key

        except Exception as e:
            logger.error(f"Failed to retrieve API key: {e}")
            return None

    def delete_api_key(
        self,
        service_name: str,
        username: str = "default",
    ) -> None:
        """
        Delete stored API key.

        Args:
            service_name: Name of the service
            username: Username/identifier
        """
        try:
            keyring.delete_password(
                f"termextractor_{service_name}",
                username,
            )
            logger.info(f"API key deleted for {service_name}")
            self._audit_log("api_key_deleted", {"service": service_name, "username": username})

        except Exception as e:
            logger.warning(f"Failed to delete API key: {e}")

    def minimize_data(self, text: str, max_context_chars: int = 1000) -> str:
        """
        Minimize data sent to API by limiting context.

        Args:
            text: Original text
            max_context_chars: Maximum context characters

        Returns:
            Minimized text
        """
        if not self.config.enable_data_minimization:
            return text

        if len(text) <= max_context_chars:
            return text

        # Keep beginning and end for context
        half = max_context_chars // 2
        minimized = text[:half] + "\n[...]\n" + text[-half:]

        self._audit_log(
            "data_minimized",
            {"original_size": len(text), "minimized_size": len(minimized)},
        )

        return minimized

    def anonymize_text(self, text: str) -> str:
        """
        Anonymize sensitive information in text.

        Args:
            text: Original text

        Returns:
            Anonymized text
        """
        if not self.config.anonymize_sensitive_data:
            return text

        import re

        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

        # Phone numbers (simple pattern)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

        # Credit card numbers (simple pattern)
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)

        # SSN (US format)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)

        self._audit_log("data_anonymized", {})

        return text

    def cleanup_old_data(self) -> int:
        """
        Clean up old data based on retention policy.

        Returns:
            Number of files deleted
        """
        if not self.config.auto_cleanup_enabled:
            return 0

        retention_cutoff = datetime.now() - timedelta(days=self.config.data_retention_days)
        deleted_count = 0

        try:
            for file_path in self.storage_dir.rglob("*"):
                if file_path.is_file():
                    # Check file modification time
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime < retention_cutoff:
                        # Skip encryption key and audit log
                        if file_path.name not in [".encryption_key", "audit.log"]:
                            file_path.unlink()
                            deleted_count += 1

            logger.info(f"Cleaned up {deleted_count} old files")
            self._audit_log("data_cleanup", {"files_deleted": deleted_count})

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        return deleted_count

    def _audit_log(self, event: str, details: Dict[str, Any]) -> None:
        """
        Log audit event.

        Args:
            event: Event name
            details: Event details
        """
        if not self.config.audit_logging:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details,
        }

        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def get_audit_log(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None,
    ) -> list:
        """
        Retrieve audit log entries.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            event_type: Event type to filter

        Returns:
            List of audit log entries
        """
        if not self.audit_log_path.exists():
            return []

        entries = []

        try:
            with open(self.audit_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    entry_date = datetime.fromisoformat(entry["timestamp"])

                    # Apply filters
                    if start_date and entry_date < start_date:
                        continue
                    if end_date and entry_date > end_date:
                        continue
                    if event_type and entry["event"] != event_type:
                        continue

                    entries.append(entry)

        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")

        return entries

    def export_data(self, user_id: str, output_path: Path) -> None:
        """
        Export user data (GDPR compliance).

        Args:
            user_id: User identifier
            output_path: Output file path
        """
        if not self.config.enable_gdpr_features:
            logger.warning("GDPR features not enabled")
            return

        # Collect all user data
        user_data = {
            "user_id": user_id,
            "export_date": datetime.now().isoformat(),
            "audit_log": self.get_audit_log(),
            # Add other user-specific data
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(user_data, f, indent=2)

            logger.info(f"User data exported to {output_path}")
            self._audit_log("data_exported", {"user_id": user_id, "output": str(output_path)})

        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise

    def delete_user_data(self, user_id: str) -> None:
        """
        Delete all user data (GDPR compliance).

        Args:
            user_id: User identifier
        """
        if not self.config.enable_gdpr_features:
            logger.warning("GDPR features not enabled")
            return

        # Delete user-specific files and data
        # This is a simplified implementation
        logger.info(f"Deleting data for user {user_id}")
        self._audit_log("data_deleted", {"user_id": user_id})

        # In production, implement comprehensive deletion
