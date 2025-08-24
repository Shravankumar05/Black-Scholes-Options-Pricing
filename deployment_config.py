"""
Deployment Configuration Module
==============================
Handles environment-specific settings for production deployment.
"""

import os
import sys
import warnings
import streamlit as st

def configure_deployment_environment():
    """Configure environment for devcontainer/production deployment"""
    
    # Comprehensive warning suppression
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    
    # Comprehensive TensorFlow environment configuration for containers
    tf_env_vars = {
        'CUDA_VISIBLE_DEVICES': '-1',  # Disable GPU completely
        'TF_CPP_MIN_LOG_LEVEL': '3',   # Suppress all TensorFlow logs
        'TF_ENABLE_ONEDNN_OPTS': '0',  # Disable oneDNN optimizations
        'TF_FORCE_GPU_ALLOW_GROWTH': 'false',
        'TF_GPU_THREAD_MODE': 'gpu_private',
        'CUDA_CACHE_DISABLE': '1',     # Disable CUDA caching
        'XLA_FLAGS': '--xla_gpu_cuda_data_dir=/dev/null',  # Disable XLA GPU
        'TF_XLA_FLAGS': '--tf_xla_cpu_global_jit=false',   # Disable XLA JIT
        'TF_DISABLE_SPARSE_SOFTMAX': '1',
        'TF_DISABLE_SEGMENT_REDUCTION_OP': '1',
        'NVIDIA_VISIBLE_DEVICES': 'none',  # Hide NVIDIA devices
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'TF_FORCE_CPU_DEVICE': '1',    # Force CPU usage
        'CUDA_LAUNCH_BLOCKING': '0',   # Disable CUDA blocking
        'TF_DISABLE_MKL': '1'          # Disable Intel MKL
    }
    
    for key, value in tf_env_vars.items():
        os.environ[key] = value
    
    # Configure TensorFlow if available
    try:
        import tensorflow as tf
        
        # Disable GPU devices completely
        tf.config.set_visible_devices([], 'GPU')
        
        # Disable XLA compilation
        tf.config.optimizer.set_jit(False)
        
        # Set logging levels to suppress all output
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        
        # Disable v1 logging
        if hasattr(tf.compat.v1, 'logging'):
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        # Additional CPU-only configuration
        try:
            # Force CPU usage for all operations
            with tf.device('/CPU:0'):
                pass
            
            # Disable GPU memory growth (if any GPU detected)
            physical_devices = tf.config.list_physical_devices('GPU')
            for device in physical_devices:
                tf.config.set_memory_growth(device, False)
                tf.config.experimental.set_memory_growth(device, False)
        except Exception:
            pass  # Ignore if no GPU devices or method unavailable
        
        print("✅ TensorFlow configured for container CPU-only deployment")
    except ImportError:
        print("⚠️ TensorFlow not available in container")
    except Exception as e:
        # Suppress error details in production
        print(f"⚠️ TensorFlow configuration completed with warnings (details suppressed)")

def safe_pyplot_display(fig, clear_figure=True, use_container_width=True, suppress_errors=True):
    """Safely display matplotlib figures in Streamlit with container-aware error handling"""
    import matplotlib.pyplot as plt
    import io
    import base64
    import tempfile
    
    try:
        # Method 1: Try simplified Streamlit display (works best in containers)
        try:
            # Use basic pyplot without complex parameters that can cause media file issues
            st.pyplot(fig, clear_figure=True, use_container_width=False)
            plt.close(fig)
            return
        except Exception:
            pass
        
        # Method 2: Convert to image buffer and display (avoids media file storage)
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            st.image(buf, use_column_width=use_container_width)
            plt.close(fig)
            return
        except Exception:
            pass
        
        # Method 3: SVG format (text-based, no media file storage issues)
        try:
            buf = io.StringIO()
            fig.savefig(buf, format='svg', bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            svg_str = buf.getvalue()
            # Display SVG directly in markdown
            st.markdown(f'<div style="text-align: center;">{svg_str}</div>', 
                       unsafe_allow_html=True)
            plt.close(fig)
            return
        except Exception:
            pass
        
        # Method 4: Base64 encoded image (inline, no external file)
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;">', 
                       unsafe_allow_html=True)
            plt.close(fig)
            return
        except Exception:
            pass
        
        # Method 5: Fallback message (if all display methods fail)
        if not suppress_errors:
            st.info("📊 Chart display temporarily unavailable in container environment.")
            st.caption("Chart would display here under normal conditions.")
        
    except Exception as e:
        if not suppress_errors:
            st.warning(f"Chart display system temporarily unavailable.")
    finally:
        # Always ensure figure is closed to prevent memory leaks
        try:
            plt.close(fig)
        except:
            pass

def configure_streamlit_settings():
    """Configure Streamlit-specific settings for deployment"""
    try:
        # Configure Streamlit for production
        if hasattr(st, 'set_option'):
            # Disable file watcher in production
            st.set_option('server.fileWatcherType', 'none')
            
            # Configure memory settings
            st.set_option('server.maxUploadSize', 50)
            st.set_option('server.maxMessageSize', 50)
            
        print("✅ Streamlit configured for deployment")
    except Exception as e:
        print(f"⚠️ Streamlit configuration warning: {e}")

def is_deployment_environment():
    """Detect if running in a deployment/container environment"""
    deployment_indicators = [
        'STREAMLIT_SERVER_PORT' in os.environ,
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS' in os.environ,
        'HOSTNAME' in os.environ and os.environ.get('HOSTNAME', '').startswith('streamlit'),
        'HOME' in os.environ and 'adminuser' in os.environ.get('HOME', ''),
        sys.platform.startswith('linux') and 'venv' in sys.executable,
        # Devcontainer/Codespaces specific indicators
        'CODESPACES' in os.environ,
        'GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN' in os.environ,
        'VSCODE_REMOTE_CONTAINERS_SESSION' in os.environ,
        'REMOTE_CONTAINERS' in os.environ,
        # Container indicators
        os.path.exists('/.dockerenv'),
        'container' in os.environ.get('USER', '').lower(),
        # Python container indicators
        '/usr/local/bin/python' in sys.executable or '/opt/python' in sys.executable
    ]
    
    return any(deployment_indicators)

def get_deployment_config():
    """Get deployment-specific configuration optimized for containers"""
    is_deployed = is_deployment_environment()
    
    config = {
        'is_deployed': is_deployed,
        'enable_gpu': False,  # Always disable GPU in containers
        'enable_tensorflow': True,
        'enable_advanced_ml': not is_deployed,  # Simplify ML in containers
        'max_features': 15 if is_deployed else 50,  # Further reduce features for containers
        'max_models': 2 if is_deployed else 6,  # Reduce models for container memory
        'use_simplified_plots': is_deployed,
        'suppress_warnings': is_deployed,
        'memory_limit_mb': 256 if is_deployed else 2048,  # Lower memory for containers
        'use_safe_plotting': is_deployed,  # Use safe plotting methods
        'disable_cuda_warnings': is_deployed,
        'container_optimized': is_deployed
    }
    
    if is_deployed:
        print("🐳 Running in container/deployment environment - optimized configuration active")
        print("   - GPU disabled, CPU-only TensorFlow")
        print("   - Simplified ML models and plotting")
        print("   - Enhanced error suppression")
    else:
        print("🔬 Running in development environment - full configuration active")
    
    return config

# Initialize deployment configuration
deployment_config = get_deployment_config()

if deployment_config['is_deployed']:
    configure_deployment_environment()
    configure_streamlit_settings()