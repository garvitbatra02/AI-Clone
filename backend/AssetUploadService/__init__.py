"""
Asset Upload Service — Dashboard API Layer

Provides REST endpoints for the asset upload dashboard:
  - Collection management (list, create, delete, stats)
  - File uploads (HTTP multipart + local path)
  - Upload previews (dry-run before committing)
  - Text / batch-text uploads
  - Directory uploads (local path)

All heavy lifting (chunking, embedding, VectorDB storage) is delegated
to the core AssetUploadService in RAGService.Data.services.
"""
