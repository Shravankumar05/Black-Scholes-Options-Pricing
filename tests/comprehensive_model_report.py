"""
Comprehensive Model Performance Report Generator
==============================================
This script generates a comprehensive report combining all testing results
with actionable insights and specific improvement recommendations.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveReportGenerator:
    def __init__(self):
        self.volatility_results = None
        self.portfolio_results = None
        self.benchmark_results = None
        self.overall_assessment = {}
        self.recommendations = {}
        
    def load_test_results(self):
        """Load all test results from JSON files"""
        
        try:
            with open('volatility_test_results.json', 'r') as f:
                self.volatility_results = json.load(f)
            print("✓ Loaded volatility test results")
        except FileNotFoundError:
            print("⚠️ Volatility test results not found")
            self.volatility_results = {}
        
        try:
            with open('portfolio_test_results.json', 'r') as f:
                self.portfolio_results = json.load(f)
            print("✓ Loaded portfolio test results")
        except FileNotFoundError:
            print("⚠️ Portfolio test results not found")
            self.portfolio_results = {}
        
        try:
            with open('benchmark_comparison_results.json', 'r') as f:
                self.benchmark_results = json.load(f)
            print("✓ Loaded benchmark comparison results")
        except FileNotFoundError:
            print("⚠️ Benchmark comparison results not found")
            self.benchmark_results = {}
    
    def analyze_volatility_performance(self):
        """Analyze volatility forecasting performance"""
        
        if not self.volatility_results:
            return {'status': 'no_data', 'assessment': 'Cannot assess - no data available'}
        
        # Extract key metrics
        r2_scores = []
        direction_accuracies = []
        rmse_values = []
        
        for test_name, result in self.volatility_results.items():
            if 'metrics' in result:
                r2_scores.append(result['metrics'].get('r2', 0))
                direction_accuracies.append(result['metrics'].get('direction_accuracy', 0))
                rmse_values.append(result['metrics'].get('rmse', float('inf')))
        
        if not r2_scores:
            return {'status': 'no_metrics', 'assessment': 'No valid metrics found'}
        
        # Calculate statistics
        avg_r2 = np.mean(r2_scores)
        avg_direction_acc = np.mean(direction_accuracies)
        avg_rmse = np.mean(rmse_values)
        r2_std = np.std(r2_scores)
        
        # Determine performance level
        if avg_r2 >= 0.25 and avg_direction_acc >= 0.60:
            performance_level = 'excellent'
            performance_score = 9
        elif avg_r2 >= 0.15 and avg_direction_acc >= 0.55:
            performance_level = 'good'
            performance_score = 7
        elif avg_r2 >= 0.05 and avg_direction_acc >= 0.50:
            performance_level = 'moderate'
            performance_score = 5
        else:
            performance_level = 'poor'
            performance_score = 2
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if avg_r2 > 0.15:
            strengths.append("Strong predictive accuracy (R² > 0.15)")
        if avg_direction_acc > 0.55:
            strengths.append("Good directional forecasting ability")
        if r2_std < 0.1:
            strengths.append("Consistent performance across scenarios")
        
        if avg_r2 < 0.1:
            weaknesses.append("Low predictive accuracy (R² < 0.1)")
        if avg_direction_acc < 0.52:
            weaknesses.append("Poor directional forecasting")
        if r2_std > 0.15:
            weaknesses.append("Inconsistent performance across scenarios")
        
        return {
            'status': 'success',
            'performance_level': performance_level,
            'performance_score': performance_score,
            'avg_r2': avg_r2,
            'avg_direction_accuracy': avg_direction_acc,
            'avg_rmse': avg_rmse,
            'r2_std': r2_std,
            'n_tests': len(r2_scores),
            'strengths': strengths,
            'weaknesses': weaknesses,
            'assessment': self._generate_volatility_assessment(performance_level, avg_r2, avg_direction_acc)
        }
    
    def analyze_portfolio_performance(self):
        """Analyze portfolio allocation performance"""
        
        if not self.portfolio_results:
            return {'status': 'no_data', 'assessment': 'Cannot assess - no data available'}
        
        # Extract key metrics
        sharpe_ratios = []
        max_drawdowns = []
        annual_returns = []
        volatilities = []
        
        for test_name, result in self.portfolio_results.items():
            if 'strategy_results' in result:
                # Get best strategy from each test
                best_strategy_name = result.get('best_strategy', '')
                if best_strategy_name and best_strategy_name in result['strategy_results']:
                    metrics = result['strategy_results'][best_strategy_name]
                    sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                    max_drawdowns.append(metrics.get('max_drawdown', 0))
                    annual_returns.append(metrics.get('annualized_return', 0))
                    volatilities.append(metrics.get('annualized_volatility', 0))
        
        if not sharpe_ratios:
            return {'status': 'no_metrics', 'assessment': 'No valid metrics found'}
        
        # Calculate statistics
        avg_sharpe = np.mean(sharpe_ratios)
        avg_max_dd = np.mean(max_drawdowns)
        avg_return = np.mean(annual_returns)
        avg_vol = np.mean(volatilities)
        sharpe_std = np.std(sharpe_ratios)
        
        # Determine performance level
        if avg_sharpe >= 1.0:
            performance_level = 'excellent'
            performance_score = 9
        elif avg_sharpe >= 0.6:
            performance_level = 'good'
            performance_score = 7
        elif avg_sharpe >= 0.3:
            performance_level = 'moderate'
            performance_score = 5
        else:
            performance_level = 'poor'
            performance_score = 2
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if avg_sharpe > 0.6:
            strengths.append("High risk-adjusted returns (Sharpe > 0.6)")
        if avg_max_dd > -0.15:
            strengths.append("Good downside protection (Max DD > -15%)")
        if sharpe_std < 0.3:
            strengths.append("Consistent performance across scenarios")
        if avg_return > 0.08:
            strengths.append("Strong absolute returns")
        
        if avg_sharpe < 0.3:
            weaknesses.append("Low risk-adjusted returns (Sharpe < 0.3)")
        if avg_max_dd < -0.25:
            weaknesses.append("High maximum drawdowns")
        if sharpe_std > 0.5:
            weaknesses.append("Inconsistent performance")
        
        return {
            'status': 'success',
            'performance_level': performance_level,
            'performance_score': performance_score,
            'avg_sharpe': avg_sharpe,
            'avg_max_drawdown': avg_max_dd,
            'avg_return': avg_return,
            'avg_volatility': avg_vol,
            'sharpe_std': sharpe_std,
            'n_tests': len(sharpe_ratios),
            'strengths': strengths,
            'weaknesses': weaknesses,
            'assessment': self._generate_portfolio_assessment(performance_level, avg_sharpe, avg_max_dd)
        }
    
    def analyze_benchmark_comparison(self):
        """Analyze benchmark comparison results"""
        
        if not self.benchmark_results:
            return {'status': 'no_data', 'assessment': 'Cannot assess - no data available'}
        
        summary = self.benchmark_results.get('summary', {})
        vol_improvement = summary.get('volatility_avg_improvement', 0)
        portfolio_improvement = summary.get('portfolio_avg_improvement', 0)
        
        # Assess improvements
        vol_assessment = 'significant' if vol_improvement > 0.05 else 'marginal' if vol_improvement > 0.01 else 'poor'
        portfolio_assessment = 'significant' if portfolio_improvement > 0.1 else 'marginal' if portfolio_improvement > 0.05 else 'poor'
        
        return {
            'status': 'success',
            'volatility_improvement': vol_improvement,
            'portfolio_improvement': portfolio_improvement,
            'volatility_assessment': vol_assessment,
            'portfolio_assessment': portfolio_assessment,
            'assessment': f"Volatility: {vol_assessment}, Portfolio: {portfolio_assessment}"
        }
    
    def _generate_volatility_assessment(self, performance_level, avg_r2, avg_direction_acc):
        """Generate detailed volatility assessment"""
        
        if performance_level == 'excellent':
            return f"Excellent volatility forecasting model with R² of {avg_r2:.3f} and {avg_direction_acc:.1%} direction accuracy. Ready for production deployment."
        elif performance_level == 'good':
            return f"Good volatility forecasting capability with R² of {avg_r2:.3f}. Model shows promise but could benefit from fine-tuning."
        elif performance_level == 'moderate':
            return f"Moderate forecasting ability with R² of {avg_r2:.3f}. Significant improvements needed before production use."
        else:
            return f"Poor forecasting performance with R² of {avg_r2:.3f}. Model requires complete redesign."
    
    def _generate_portfolio_assessment(self, performance_level, avg_sharpe, avg_max_dd):
        """Generate detailed portfolio assessment"""
        
        if performance_level == 'excellent':
            return f"Excellent portfolio strategies with average Sharpe ratio of {avg_sharpe:.3f}. Ready for production deployment."
        elif performance_level == 'good':
            return f"Good portfolio performance with Sharpe ratio of {avg_sharpe:.3f}. Suitable for production with monitoring."
        elif performance_level == 'moderate':
            return f"Moderate portfolio performance with Sharpe ratio of {avg_sharpe:.3f}. Improvements recommended before deployment."
        else:
            return f"Poor portfolio performance with Sharpe ratio of {avg_sharpe:.3f}. Strategies need significant enhancement."
    
    def generate_specific_recommendations(self):
        """Generate specific, actionable recommendations"""
        
        vol_analysis = self.overall_assessment.get('volatility', {})
        portfolio_analysis = self.overall_assessment.get('portfolio', {})
        benchmark_analysis = self.overall_assessment.get('benchmark', {})
        
        recommendations = {
            'volatility': [],
            'portfolio': [],
            'general': [],
            'priority': 'high'  # high, medium, low
        }
        
        # Volatility recommendations
        if vol_analysis.get('performance_level') == 'poor':
            recommendations['volatility'].extend([
                "CRITICAL: Complete model redesign required",
                "Implement LSTM/GRU neural networks for temporal dependencies",
                "Add external features (VIX, market regime indicators)",
                "Increase training data quantity and quality",
                "Consider ensemble methods combining multiple approaches"
            ])
            recommendations['priority'] = 'high'
        elif vol_analysis.get('performance_level') == 'moderate':
            recommendations['volatility'].extend([
                "Enhance feature engineering with market microstructure data",
                "Implement hyperparameter optimization",
                "Add volatility regime detection",
                "Consider factor-based volatility models"
            ])
        elif vol_analysis.get('performance_level') in ['good', 'excellent']:
            recommendations['volatility'].extend([
                "Implement model monitoring and drift detection",
                "Add confidence intervals to predictions",
                "Set up automated retraining pipeline",
                "Consider production deployment"
            ])
        
        # Portfolio recommendations
        if portfolio_analysis.get('performance_level') == 'poor':
            recommendations['portfolio'].extend([
                "CRITICAL: Review optimization algorithms",
                "Implement robust optimization techniques",
                "Add transaction cost modeling",
                "Consider multi-period optimization",
                "Improve risk model accuracy"
            ])
            recommendations['priority'] = 'high'
        elif portfolio_analysis.get('performance_level') == 'moderate':
            recommendations['portfolio'].extend([
                "Add regime-aware allocation strategies",
                "Implement factor-based portfolio construction",
                "Enhance rebalancing rules",
                "Add alternative risk measures (CVaR, etc.)"
            ])
        elif portfolio_analysis.get('performance_level') in ['good', 'excellent']:
            recommendations['portfolio'].extend([
                "Optimize rebalancing frequency",
                "Add real-time risk monitoring",
                "Implement strategy ensembles",
                "Consider production deployment"
            ])
        
        # General recommendations based on benchmark comparison
        if benchmark_analysis.get('status') == 'success':
            vol_improvement = benchmark_analysis.get('volatility_improvement', 0)
            portfolio_improvement = benchmark_analysis.get('portfolio_improvement', 0)
            
            if vol_improvement <= 0:
                recommendations['general'].append("Volatility models don't beat simple baselines - fundamental review needed")
            if portfolio_improvement <= 0:
                recommendations['general'].append("Portfolio strategies underperform benchmarks - algorithm review required")
        
        # Add data quality recommendations
        recommendations['general'].extend([
            "Implement comprehensive data quality checks",
            "Add real-time market data feeds for better accuracy",
            "Consider alternative data sources (sentiment, macro indicators)",
            "Implement proper backtesting framework with walk-forward validation"
        ])
        
        self.recommendations = recommendations
        return recommendations
    
    def create_executive_summary(self):
        """Create executive summary of all results"""
        
        vol_analysis = self.overall_assessment.get('volatility', {})
        portfolio_analysis = self.overall_assessment.get('portfolio', {})
        benchmark_analysis = self.overall_assessment.get('benchmark', {})
        
        # Overall system score
        vol_score = vol_analysis.get('performance_score', 0)
        portfolio_score = portfolio_analysis.get('performance_score', 0)
        overall_score = (vol_score + portfolio_score) / 2
        
        if overall_score >= 8:
            overall_rating = "EXCELLENT"
            deployment_status = "Ready for production deployment"
        elif overall_score >= 6:
            overall_rating = "GOOD"
            deployment_status = "Ready for production with monitoring"
        elif overall_score >= 4:
            overall_rating = "MODERATE"
            deployment_status = "Requires improvements before production"
        else:
            overall_rating = "POOR"
            deployment_status = "Significant development required"
        
        summary = {
            'overall_rating': overall_rating,
            'overall_score': overall_score,
            'deployment_status': deployment_status,
            'volatility_rating': vol_analysis.get('performance_level', 'unknown').upper(),
            'portfolio_rating': portfolio_analysis.get('performance_level', 'unknown').upper(),
            'key_strengths': [],
            'key_weaknesses': [],
            'top_priorities': []
        }
        
        # Combine strengths and weaknesses
        if vol_analysis.get('strengths'):
            summary['key_strengths'].extend([f"Volatility: {s}" for s in vol_analysis['strengths'][:2]])
        if portfolio_analysis.get('strengths'):
            summary['key_strengths'].extend([f"Portfolio: {s}" for s in portfolio_analysis['strengths'][:2]])
        
        if vol_analysis.get('weaknesses'):
            summary['key_weaknesses'].extend([f"Volatility: {s}" for s in vol_analysis['weaknesses'][:2]])
        if portfolio_analysis.get('weaknesses'):
            summary['key_weaknesses'].extend([f"Portfolio: {s}" for s in portfolio_analysis['weaknesses'][:2]])
        
        # Top priorities from recommendations
        if self.recommendations:
            if self.recommendations['priority'] == 'high':
                summary['top_priorities'].extend(self.recommendations['volatility'][:2])
                summary['top_priorities'].extend(self.recommendations['portfolio'][:2])
            else:
                summary['top_priorities'].extend(self.recommendations['general'][:3])
        
        return summary
    
    def generate_final_report(self):
        """Generate the complete final report"""
        
        print("\n" + "="*100)
        print("📋 COMPREHENSIVE MODEL PERFORMANCE EVALUATION REPORT")
        print("="*100)
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load and analyze all results
        self.load_test_results()
        
        self.overall_assessment['volatility'] = self.analyze_volatility_performance()
        self.overall_assessment['portfolio'] = self.analyze_portfolio_performance()
        self.overall_assessment['benchmark'] = self.analyze_benchmark_comparison()
        
        # Generate recommendations
        self.generate_specific_recommendations()
        
        # Create executive summary
        summary = self.create_executive_summary()
        
        # Print Executive Summary
        print(f"\n🎯 EXECUTIVE SUMMARY")
        print("="*50)
        print(f"Overall System Rating: {summary['overall_rating']} ({summary['overall_score']:.1f}/10)")
        print(f"Deployment Status: {summary['deployment_status']}")
        print(f"Volatility Forecasting: {summary['volatility_rating']}")
        print(f"Portfolio Allocation: {summary['portfolio_rating']}")
        
        # Key Findings
        print(f"\n🔍 KEY FINDINGS")
        print("-"*30)
        
        # Volatility Analysis
        vol_analysis = self.overall_assessment['volatility']
        if vol_analysis['status'] == 'success':
            print(f"\n📊 Volatility Forecasting:")
            print(f"  Performance Level: {vol_analysis['performance_level'].upper()}")
            print(f"  Average R² Score: {vol_analysis['avg_r2']:.4f}")
            print(f"  Direction Accuracy: {vol_analysis['avg_direction_accuracy']:.2%}")
            print(f"  Tests Conducted: {vol_analysis['n_tests']}")
            print(f"  Assessment: {vol_analysis['assessment']}")
        
        # Portfolio Analysis
        portfolio_analysis = self.overall_assessment['portfolio']
        if portfolio_analysis['status'] == 'success':
            print(f"\n📈 Portfolio Allocation:")
            print(f"  Performance Level: {portfolio_analysis['performance_level'].upper()}")
            print(f"  Average Sharpe Ratio: {portfolio_analysis['avg_sharpe']:.4f}")
            print(f"  Average Max Drawdown: {portfolio_analysis['avg_max_drawdown']:.2%}")
            print(f"  Tests Conducted: {portfolio_analysis['n_tests']}")
            print(f"  Assessment: {portfolio_analysis['assessment']}")
        
        # Benchmark Comparison
        benchmark_analysis = self.overall_assessment['benchmark']
        if benchmark_analysis['status'] == 'success':
            print(f"\n🏆 Benchmark Comparison:")
            print(f"  Volatility vs Benchmarks: {benchmark_analysis['volatility_assessment'].upper()}")
            print(f"  Portfolio vs Benchmarks: {benchmark_analysis['portfolio_assessment'].upper()}")
            print(f"  Volatility Improvement: {benchmark_analysis['volatility_improvement']:+.4f}")
            print(f"  Portfolio Improvement: {benchmark_analysis['portfolio_improvement']:+.4f}")
        
        # Strengths and Weaknesses
        if summary['key_strengths']:
            print(f"\n✅ KEY STRENGTHS:")
            for strength in summary['key_strengths'][:5]:
                print(f"  • {strength}")
        
        if summary['key_weaknesses']:
            print(f"\n⚠️  KEY WEAKNESSES:")
            for weakness in summary['key_weaknesses'][:5]:
                print(f"  • {weakness}")
        
        # Detailed Recommendations
        print(f"\n💡 ACTIONABLE RECOMMENDATIONS")
        print("="*50)
        
        print(f"\n🔧 VOLATILITY MODEL IMPROVEMENTS:")
        for i, rec in enumerate(self.recommendations['volatility'][:5], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n📊 PORTFOLIO STRATEGY IMPROVEMENTS:")
        for i, rec in enumerate(self.recommendations['portfolio'][:5], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n🌟 GENERAL RECOMMENDATIONS:")
        for i, rec in enumerate(self.recommendations['general'][:5], 1):
            print(f"  {i}. {rec}")
        
        # Priority Actions
        if summary['top_priorities']:
            print(f"\n🚨 TOP PRIORITY ACTIONS:")
            for i, priority in enumerate(summary['top_priorities'][:3], 1):
                print(f"  {i}. {priority}")
        
        # Implementation Roadmap
        print(f"\n🗺️  IMPLEMENTATION ROADMAP")
        print("-"*40)
        
        if summary['overall_rating'] in ['POOR', 'MODERATE']:
            print("Phase 1 (Immediate - 2-4 weeks):")
            print("  • Address critical model deficiencies")
            print("  • Implement basic improvements")
            print("  • Enhance data quality and preprocessing")
            print("\nPhase 2 (Short-term - 1-2 months):")
            print("  • Implement recommended algorithm improvements")
            print("  • Add comprehensive testing framework")
            print("  • Begin limited testing with small capital")
            print("\nPhase 3 (Medium-term - 3-6 months):")
            print("  • Full production deployment")
            print("  • Monitoring and alerting systems")
            print("  • Continuous model improvement")
        else:
            print("Phase 1 (Immediate - 1-2 weeks):")
            print("  • Set up production monitoring")
            print("  • Implement risk management controls")
            print("  • Begin phased deployment")
            print("\nPhase 2 (Short-term - 1 month):")
            print("  • Full production deployment")
            print("  • Performance tracking and optimization")
            print("  • Expand to additional asset classes")
        
        # Risk Assessment
        print(f"\n⚠️  RISK ASSESSMENT")
        print("-"*30)
        
        risk_level = "HIGH" if summary['overall_score'] < 5 else "MEDIUM" if summary['overall_score'] < 7 else "LOW"
        print(f"Deployment Risk Level: {risk_level}")
        
        if risk_level == "HIGH":
            print("  • Models may perform poorly in live trading")
            print("  • Significant capital at risk")
            print("  • Recommend extensive improvements before deployment")
        elif risk_level == "MEDIUM":
            print("  • Models show promise but need monitoring")
            print("  • Start with limited capital allocation")
            print("  • Implement strict risk controls")
        else:
            print("  • Models ready for production deployment")
            print("  • Standard risk management sufficient")
            print("  • Monitor for model drift")
        
        # Save complete report
        self._save_complete_report(summary)
        
        print(f"\n💾 Complete report saved to 'comprehensive_model_report.json'")
        print("="*100)
        
        return summary
    
    def _save_complete_report(self, summary):
        """Save the complete report to file"""
        
        report_data = {
            'generation_date': datetime.now().isoformat(),
            'executive_summary': summary,
            'detailed_analysis': self.overall_assessment,
            'recommendations': self.recommendations,
            'raw_results': {
                'volatility': self.volatility_results,
                'portfolio': self.portfolio_results,
                'benchmark': self.benchmark_results
            }
        }
        
        with open('comprehensive_model_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

def main():
    """Main execution function"""
    reporter = ComprehensiveReportGenerator()
    summary = reporter.generate_final_report()
    return summary

if __name__ == "__main__":
    main()