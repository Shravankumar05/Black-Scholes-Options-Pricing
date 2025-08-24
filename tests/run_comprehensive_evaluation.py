"""
Master Test Runner for ML Models Evaluation
==========================================
This script runs all comprehensive tests for volatility forecasting and portfolio allocation,
generates benchmark comparisons, and produces a final evaluation report with specific
recommendations for improvement.
"""

import sys
import time
from datetime import datetime
import traceback
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import all test modules
try:
    from test_ml_volatility_comprehensive import VolatilityTestFramework
    from test_portfolio_comprehensive import PortfolioTestFramework
    from test_model_comparison import ModelBenchmarkFramework
    from comprehensive_model_report import ComprehensiveReportGenerator
    print("✓ All test modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import test modules: {e}")
    sys.exit(1)

class MasterTestRunner:
    def __init__(self):
        self.start_time = None
        self.results_summary = {}
        self.test_status = {
            'volatility_tests': 'pending',
            'portfolio_tests': 'pending',
            'benchmark_comparison': 'pending',
            'final_report': 'pending'
        }
    
    def print_header(self):
        """Print the main header"""
        print("\n" + "="*120)
        print("🚀 COMPREHENSIVE ML MODELS EVALUATION SUITE")
        print("="*120)
        print("This comprehensive evaluation will test:")
        print("  📊 ML Volatility Forecasting Model - Accuracy, backtesting, performance metrics")
        print("  📈 Portfolio Allocation Strategies - Risk-adjusted returns, strategy comparison")
        print("  🏆 Benchmark Comparisons - Performance vs industry standards")
        print("  📋 Final Report Generation - Actionable insights and recommendations")
        print()
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*120)
    
    def run_volatility_tests(self):
        """Run comprehensive volatility forecasting tests"""
        
        print("\n" + "🔬 PHASE 1: VOLATILITY FORECASTING EVALUATION")
        print("="*70)
        
        try:
            self.test_status['volatility_tests'] = 'running'
            
            print("🚀 Initializing volatility test framework...")
            tester = VolatilityTestFramework()
            
            print("📊 Running comprehensive evaluation...")
            print("   This includes:")
            print("     • Multiple market scenarios (normal, crisis, low-vol, regime changes)")
            print("     • Real market data testing")
            print("     • Walk-forward backtesting")
            print("     • Baseline comparisons")
            print("     • Statistical significance testing")
            
            # Run the comprehensive evaluation
            tester.run_comprehensive_evaluation()
            
            self.test_status['volatility_tests'] = 'completed'
            self.results_summary['volatility'] = {
                'status': 'success',
                'message': 'Volatility forecasting tests completed successfully',
                'output_file': 'volatility_test_results.json'
            }
            
            print("✅ Volatility forecasting evaluation completed successfully!")
            
        except Exception as e:
            self.test_status['volatility_tests'] = 'failed'
            self.results_summary['volatility'] = {
                'status': 'error',
                'message': f'Volatility tests failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
            print(f"❌ Volatility tests failed: {e}")
            print("Continuing with other tests...")
    
    def run_portfolio_tests(self):
        """Run comprehensive portfolio allocation tests"""
        
        print("\n" + "📈 PHASE 2: PORTFOLIO ALLOCATION EVALUATION")
        print("="*70)
        
        try:
            self.test_status['portfolio_tests'] = 'running'
            
            print("🚀 Initializing portfolio test framework...")
            tester = PortfolioTestFramework()
            
            print("📊 Running comprehensive evaluation...")
            print("   This includes:")
            print("     • Multiple allocation strategies testing")
            print("     • Risk-adjusted performance metrics")
            print("     • Market scenario stress testing")
            print("     • Strategy comparison and ranking")
            print("     • Transaction cost analysis")
            
            # Run the comprehensive evaluation
            tester.run_comprehensive_evaluation()
            
            self.test_status['portfolio_tests'] = 'completed'
            self.results_summary['portfolio'] = {
                'status': 'success',
                'message': 'Portfolio allocation tests completed successfully',
                'output_file': 'portfolio_test_results.json'
            }
            
            print("✅ Portfolio allocation evaluation completed successfully!")
            
        except Exception as e:
            self.test_status['portfolio_tests'] = 'failed'
            self.results_summary['portfolio'] = {
                'status': 'error',
                'message': f'Portfolio tests failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
            print(f"❌ Portfolio tests failed: {e}")
            print("Continuing with other tests...")
    
    def run_benchmark_comparison(self):
        """Run benchmark comparison tests"""
        
        print("\n" + "🏆 PHASE 3: BENCHMARK COMPARISON ANALYSIS")
        print("="*70)
        
        try:
            self.test_status['benchmark_comparison'] = 'running'
            
            print("🚀 Initializing benchmark comparison framework...")
            benchmarker = ModelBenchmarkFramework()
            
            print("📊 Running comprehensive benchmark comparison...")
            print("   This includes:")
            print("     • Comparison against industry-standard baselines")
            print("     • Statistical significance testing")
            print("     • Performance visualization")
            print("     • Relative performance assessment")
            
            # Run the comprehensive benchmark comparison
            benchmarker.run_comprehensive_benchmark_comparison()
            
            self.test_status['benchmark_comparison'] = 'completed'
            self.results_summary['benchmark'] = {
                'status': 'success',
                'message': 'Benchmark comparison completed successfully',
                'output_file': 'benchmark_comparison_results.json'
            }
            
            print("✅ Benchmark comparison completed successfully!")
            
        except Exception as e:
            self.test_status['benchmark_comparison'] = 'failed'
            self.results_summary['benchmark'] = {
                'status': 'error',
                'message': f'Benchmark comparison failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
            print(f"❌ Benchmark comparison failed: {e}")
            print("Continuing with final report generation...")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        
        print("\n" + "📋 PHASE 4: COMPREHENSIVE REPORT GENERATION")
        print("="*70)
        
        try:
            self.test_status['final_report'] = 'running'
            
            print("🚀 Initializing report generator...")
            reporter = ComprehensiveReportGenerator()
            
            print("📊 Generating comprehensive final report...")
            print("   This includes:")
            print("     • Executive summary with overall assessment")
            print("     • Detailed performance analysis")
            print("     • Specific actionable recommendations")
            print("     • Implementation roadmap")
            print("     • Risk assessment")
            
            # Generate the final report
            summary = reporter.generate_final_report()
            
            self.test_status['final_report'] = 'completed'
            self.results_summary['final_report'] = {
                'status': 'success',
                'message': 'Final report generated successfully',
                'output_file': 'comprehensive_model_report.json',
                'overall_rating': summary.get('overall_rating', 'Unknown'),
                'deployment_status': summary.get('deployment_status', 'Unknown')
            }
            
            print("✅ Comprehensive final report generated successfully!")
            
            return summary
            
        except Exception as e:
            self.test_status['final_report'] = 'failed'
            self.results_summary['final_report'] = {
                'status': 'error',
                'message': f'Final report generation failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
            print(f"❌ Final report generation failed: {e}")
            return None
    
    def print_execution_summary(self, final_summary=None):
        """Print final execution summary"""
        
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        print("\n" + "="*120)
        print("📋 EXECUTION SUMMARY")
        print("="*120)
        
        print(f"⏰ Total Execution Time: {total_duration}")
        print(f"🏁 Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📊 TEST PHASE RESULTS:")
        
        # Status symbols
        status_symbols = {
            'completed': '✅',
            'failed': '❌',
            'running': '⏳',
            'pending': '⏸️'
        }
        
        for phase, status in self.test_status.items():
            symbol = status_symbols.get(status, '❓')
            phase_name = phase.replace('_', ' ').title()
            print(f"  {symbol} {phase_name}: {status.upper()}")
        
        # Results summary
        print(f"\n📁 GENERATED FILES:")
        for test_name, result in self.results_summary.items():
            if result['status'] == 'success' and 'output_file' in result:
                print(f"  📄 {result['output_file']} - {result['message']}")
        
        # Final assessment if available
        if final_summary:
            print(f"\n🎯 FINAL ASSESSMENT:")
            print(f"  Overall Rating: {final_summary.get('overall_rating', 'Unknown')}")
            print(f"  Deployment Status: {final_summary.get('deployment_status', 'Unknown')}")
            print(f"  Volatility Model: {final_summary.get('volatility_rating', 'Unknown')}")
            print(f"  Portfolio Strategies: {final_summary.get('portfolio_rating', 'Unknown')}")
        
        # Error summary
        errors = [result for result in self.results_summary.values() if result['status'] == 'error']
        if errors:
            print(f"\n⚠️  ERRORS ENCOUNTERED:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error['message']}")
        
        # Success rate
        total_phases = len(self.test_status)
        completed_phases = sum(1 for status in self.test_status.values() if status == 'completed')
        success_rate = (completed_phases / total_phases) * 100
        
        print(f"\n📈 SUCCESS RATE: {completed_phases}/{total_phases} phases completed ({success_rate:.1f}%)")
        
        # Next steps
        print(f"\n🚀 NEXT STEPS:")
        if success_rate == 100:
            print("  1. Review the comprehensive model report")
            print("  2. Implement recommended improvements")
            print("  3. Consider production deployment based on assessment")
        elif success_rate >= 75:
            print("  1. Review completed test results")
            print("  2. Investigate and resolve failed phases")
            print("  3. Re-run failed tests after fixes")
        else:
            print("  1. Check error logs and resolve issues")
            print("  2. Ensure all dependencies are properly installed")
            print("  3. Re-run the complete test suite")
        
        print("="*120)
    
    def run_complete_evaluation(self):
        """Run the complete evaluation suite"""
        
        self.start_time = datetime.now()
        self.print_header()
        
        print("🔥 Starting comprehensive model evaluation...")
        print("   Estimated completion time: 15-30 minutes")
        print("   Please be patient as tests run extensive backtesting and analysis")
        
        # Phase 1: Volatility Tests
        self.run_volatility_tests()
        time.sleep(2)  # Brief pause between phases
        
        # Phase 2: Portfolio Tests  
        self.run_portfolio_tests()
        time.sleep(2)
        
        # Phase 3: Benchmark Comparison
        self.run_benchmark_comparison()
        time.sleep(2)
        
        # Phase 4: Final Report
        final_summary = self.generate_final_report()
        
        # Print execution summary
        self.print_execution_summary(final_summary)
        
        return final_summary

def main():
    """Main execution function"""
    
    print("🚀 Initializing Master Test Runner for ML Models Evaluation...")
    
    try:
        runner = MasterTestRunner()
        final_summary = runner.run_complete_evaluation()
        
        # Print final message
        if final_summary:
            overall_rating = final_summary.get('overall_rating', 'Unknown')
            if overall_rating in ['EXCELLENT', 'GOOD']:
                print("\n🎉 CONGRATULATIONS! Your models show strong performance!")
                print("   Consider proceeding with production deployment.")
            elif overall_rating == 'MODERATE':
                print("\n⚠️  Your models show promise but need improvements.")
                print("   Review recommendations before production deployment.")
            else:
                print("\n🔧 Your models need significant improvements.")
                print("   Focus on the recommended changes before deployment.")
        
        print("\n📧 For questions or support, refer to the generated documentation.")
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user.")
        print("You can re-run this script to continue the evaluation.")
        return False
        
    except Exception as e:
        print(f"\n❌ Critical error during evaluation: {e}")
        print("Please check the error logs and try again.")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)