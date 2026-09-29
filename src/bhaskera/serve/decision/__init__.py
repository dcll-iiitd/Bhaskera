"""
bhaskera.serve.decision
=======================
Typed decisions read from a model's answer-token logits (no generation), served through
Ray Serve. Ported from feder-cr/jev (MIT, see NOTICE) and extended for Bhaskera.

Selected with ``serve.backend: decision``; see ``configs/serve_jevos.yaml``.
"""
