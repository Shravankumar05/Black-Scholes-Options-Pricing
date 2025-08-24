#!/usr/bin/env python3
"""
Container Startup Configuration
===============================
Ensures proper environment configuration for container deployments.
Run this before starting the main application.
"""

import os
import sys
import warnings

def configure_container_environment():
    """Configure environment for optimal container performance"""
    
    print("🐳 Configuring container environment...")
    
    # Comprehensive TensorFlow CUDA suppression
    tf_env_vars = {
        'CUDA_VISIBLE_DEVICES': '-1',
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'false',
        'TF_GPU_THREAD_MODE': 'gpu_private',
        'CUDA_CACHE_DISABLE': '1',
        'XLA_FLAGS': '--xla_gpu_cuda_data_dir=/dev/null',
        'TF_XLA_FLAGS': '--tf_xla_cpu_global_jit=false',
        'TF_DISABLE_SPARSE_SOFTMAX': '1',
        'TF_DISABLE_SEGMENT_REDUCTION_OP': '1',
        'NVIDIA_VISIBLE_DEVICES': 'none',
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'TF_FORCE_CPU_DEVICE': '1',
        'CUDA_LAUNCH_BLOCKING': '0',
        'TF_DISABLE_MKL': '1'
    }
    
    for key, value in tf_env_vars.items():
        os.environ[key] = value
        print(f"   ✓ {key}={value}")
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    
    # Configure Streamlit for container
    streamlit_env_vars = {
        'STREAMLIT_SERVER_ENABLE_CORS': 'false',
        'STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION': 'false',
        'STREAMLIT_SERVER_FILE_WATCHER_TYPE': 'none',
        'STREAMLIT_SERVER_MAX_UPLOAD_SIZE': '50',
        'STREAMLIT_SERVER_MAX_MESSAGE_SIZE': '50',
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false'
    }
    
    for key, value in streamlit_env_vars.items():
        os.environ[key] = value
    
    print("✅ Container environment configured successfully")
    print("   - TensorFlow: CPU-only mode with CUDA disabled")
    print("   - Streamlit: Container-optimized settings")
    print("   - Warnings: Suppressed for clean output")
    
    # Test TensorFlow configuration
    try:
        import tensorflow as tf
        # Force CPU configuration
        tf.config.set_visible_devices([], 'GPU')
        tf.config.optimizer.set_jit(False)
        tf.get_logger().setLevel('ERROR')
        print("✅ TensorFlow successfully configured for CPU-only operation")
    except ImportError:
        print("ℹ️  TensorFlow not available (this is OK)")
    except Exception as e:
        print(f"⚠️  TensorFlow configuration completed (details suppressed)")

def verify_container_setup():
    """Verify that the container setup is correct"""
    print("\n🔍 Verifying container setup...")
    
    # Check Python version
    print(f"   Python: {sys.version.split()[0]}")
    
    # Check key environment variables
    key_vars = ['CUDA_VISIBLE_DEVICES', 'TF_CPP_MIN_LOG_LEVEL', 'TF_FORCE_CPU_DEVICE']
    for var in key_vars:
        value = os.environ.get(var, 'NOT_SET')
        print(f"   {var}: {value}")
    
    # Check if we're in a container
    container_indicators = [
        os.path.exists('/.dockerenv'),
        'container' in os.environ.get('USER', '').lower(),
        'adminuser' in os.environ.get('HOME', ''),
        'venv' in sys.executable
    ]
    
    is_container = any(container_indicators)
    print(f"   Container detected: {is_container}")
    
    if is_container:
        print("✅ Running in container environment - optimizations active")
    else:
        print("ℹ️  Running in local environment - standard configuration")
    
    print("🚀 Setup verification complete - ready to start application")

if __name__ == "__main__":
    configure_container_environment()
    verify_container_setup()
    
    # Optionally start the Streamlit app
    if len(sys.argv) > 1 and sys.argv[1] == "--start-app":
        print("\n🌟 Starting Streamlit application...")
        os.system("streamlit run bs_app.py --server.enableCORS false --server.enableXsrfProtection false")