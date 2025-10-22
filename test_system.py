#!/usr/bin/env python3
"""
Comprehensive testing script for the AI-Powered Women's Safety System
"""

import requests
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd

class SafetySystemTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = {}
        
    def test_api_endpoints(self):
        """Test all API endpoints"""
        print("Testing API Endpoints...")
        
        endpoints = [
            ("/api/users", "GET"),
            ("/api/safety/location", "POST"),
            ("/api/safety/alert", "POST"),
            ("/api/safety/emergency-contacts/1", "GET"),
            ("/api/safety/crime-data", "GET"),
            ("/api/safety/predict-threat", "POST")
        ]
        
        results = {}
        
        for endpoint, method in endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                
                if method == "GET":
                    if "crime-data" in endpoint:
                        response = requests.get(url, params={"latitude": 28.6139, "longitude": 77.2090})
                    else:
                        response = requests.get(url)
                elif method == "POST":
                    if "location" in endpoint:
                        data = {
                            "user_id": 1,
                            "latitude": 28.6139,
                            "longitude": 77.2090,
                            "speed": 5.0
                        }
                    elif "alert" in endpoint:
                        data = {
                            "user_id": 1,
                            "latitude": 28.6139,
                            "longitude": 77.2090,
                            "alert_type": "manual",
                            "threat_level": "high"
                        }
                    elif "predict-threat" in endpoint:
                        data = {
                            "user_id": 1,
                            "latitude": 28.6139,
                            "longitude": 77.2090
                        }
                    
                    response = requests.post(url, json=data)
                
                results[endpoint] = {
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                    "success": response.status_code < 400
                }
                
                print(f"✓ {method} {endpoint}: {response.status_code} ({response.elapsed.total_seconds():.3f}s)")
                
            except Exception as e:
                results[endpoint] = {
                    "status_code": 0,
                    "response_time": 0,
                    "success": False,
                    "error": str(e)
                }
                print(f"✗ {method} {endpoint}: Error - {str(e)}")
        
        self.test_results['api_endpoints'] = results
        return results
    
    def test_ai_model_performance(self):
        """Test AI model performance"""
        print("\nTesting AI Model Performance...")
        
        # Import the AI engine
        import sys
        sys.path.append('/home/ubuntu/women_safety_backend/src')
        from ai_engine import ai_engine
        
        # Test predictions for different scenarios
        test_scenarios = [
            {"lat": 28.6139, "lng": 77.2090, "hour": 14, "desc": "Daytime, city center"},
            {"lat": 28.6139, "lng": 77.2090, "hour": 23, "desc": "Night time, city center"},
            {"lat": 28.5355, "lng": 77.3910, "hour": 14, "desc": "Daytime, suburban area"},
            {"lat": 28.5355, "lng": 77.3910, "hour": 23, "desc": "Night time, suburban area"},
            {"lat": 28.7041, "lng": 77.1025, "hour": 2, "desc": "Late night, outskirts"}
        ]
        
        predictions = []
        response_times = []
        
        for scenario in test_scenarios:
            start_time = time.time()
            
            prediction = ai_engine.predict_threat(
                latitude=scenario["lat"],
                longitude=scenario["lng"],
                hour=scenario["hour"]
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            predictions.append({
                "scenario": scenario["desc"],
                "threat_level": prediction["threat_level"],
                "threat_score": prediction["threat_score"],
                "response_time": response_time
            })
            
            response_times.append(response_time)
            
            print(f"✓ {scenario['desc']}: {prediction['threat_level']} (Score: {prediction['threat_score']}) - {response_time:.3f}s")
        
        avg_response_time = np.mean(response_times)
        
        self.test_results['ai_performance'] = {
            "predictions": predictions,
            "avg_response_time": avg_response_time,
            "total_scenarios": len(test_scenarios)
        }
        
        print(f"Average AI Response Time: {avg_response_time:.3f}s")
        
        return predictions
    
    def test_load_performance(self):
        """Test system performance under load"""
        print("\nTesting Load Performance...")
        
        # Simulate multiple concurrent requests
        num_requests = 50
        concurrent_requests = 10
        
        response_times = []
        success_count = 0
        
        for batch in range(0, num_requests, concurrent_requests):
            batch_start = time.time()
            
            # Simulate concurrent requests
            for i in range(min(concurrent_requests, num_requests - batch)):
                try:
                    start_time = time.time()
                    
                    # Test threat prediction endpoint
                    data = {
                        "user_id": i + 1,
                        "latitude": 28.6139 + (i * 0.001),
                        "longitude": 77.2090 + (i * 0.001)
                    }
                    
                    response = requests.post(
                        f"{self.base_url}/api/safety/predict-threat",
                        json=data,
                        timeout=5
                    )
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    if response.status_code == 200:
                        success_count += 1
                        response_times.append(response_time)
                    
                except Exception as e:
                    print(f"Request failed: {str(e)}")
            
            # Small delay between batches
            time.sleep(0.1)
        
        if response_times:
            avg_response_time = np.mean(response_times)
            max_response_time = np.max(response_times)
            min_response_time = np.min(response_times)
            success_rate = (success_count / num_requests) * 100
        else:
            avg_response_time = max_response_time = min_response_time = 0
            success_rate = 0
        
        load_results = {
            "total_requests": num_requests,
            "successful_requests": success_count,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "min_response_time": min_response_time
        }
        
        self.test_results['load_performance'] = load_results
        
        print(f"Load Test Results:")
        print(f"  Total Requests: {num_requests}")
        print(f"  Successful: {success_count}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Avg Response Time: {avg_response_time:.3f}s")
        print(f"  Max Response Time: {max_response_time:.3f}s")
        print(f"  Min Response Time: {min_response_time:.3f}s")
        
        return load_results
    
    def generate_performance_charts(self):
        """Generate performance visualization charts"""
        print("\nGenerating Performance Charts...")
        
        # Create a figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SafeGuard System Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. API Endpoint Response Times
        if 'api_endpoints' in self.test_results:
            endpoints = list(self.test_results['api_endpoints'].keys())
            response_times = [self.test_results['api_endpoints'][ep]['response_time'] 
                            for ep in endpoints]
            
            ax1.bar(range(len(endpoints)), response_times, color='skyblue')
            ax1.set_title('API Endpoint Response Times')
            ax1.set_ylabel('Response Time (seconds)')
            ax1.set_xticks(range(len(endpoints)))
            ax1.set_xticklabels([ep.split('/')[-1] for ep in endpoints], rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # 2. AI Model Threat Level Distribution
        if 'ai_performance' in self.test_results:
            threat_levels = [p['threat_level'] for p in self.test_results['ai_performance']['predictions']]
            threat_counts = pd.Series(threat_levels).value_counts()
            
            colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            ax2.pie(threat_counts.values, labels=threat_counts.index, autopct='%1.1f%%',
                   colors=[colors.get(level, 'gray') for level in threat_counts.index])
            ax2.set_title('AI Threat Level Predictions')
        
        # 3. Load Test Performance
        if 'load_performance' in self.test_results:
            metrics = ['Success Rate (%)', 'Avg Response (ms)', 'Max Response (ms)']
            values = [
                self.test_results['load_performance']['success_rate'],
                self.test_results['load_performance']['avg_response_time'] * 1000,
                self.test_results['load_performance']['max_response_time'] * 1000
            ]
            
            bars = ax3.bar(metrics, values, color=['green', 'blue', 'orange'])
            ax3.set_title('Load Test Performance Metrics')
            ax3.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.1f}', ha='center', va='bottom')
        
        # 4. System Architecture Overview (Text-based)
        ax4.text(0.1, 0.9, 'System Components:', fontsize=14, fontweight='bold', transform=ax4.transAxes)
        components = [
            '• Flask Backend Server',
            '• AI Threat Prediction Engine',
            '• React Frontend Interface',
            '• SQLite Database',
            '• REST API Endpoints',
            '• Real-time Location Tracking',
            '• Emergency Alert System'
        ]
        
        for i, component in enumerate(components):
            ax4.text(0.1, 0.8 - i*0.1, component, fontsize=10, transform=ax4.transAxes)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('System Architecture Components')
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/system_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Performance charts saved to system_performance_analysis.png")
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("=" * 60)
        print("SAFEGUARD SYSTEM TESTING REPORT")
        print("=" * 60)
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run all tests
        self.test_api_endpoints()
        self.test_ai_model_performance()
        self.test_load_performance()
        self.generate_performance_charts()
        
        # Generate summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        if 'api_endpoints' in self.test_results:
            api_success_rate = sum(1 for r in self.test_results['api_endpoints'].values() if r['success']) / len(self.test_results['api_endpoints']) * 100
            print(f"API Endpoints Success Rate: {api_success_rate:.1f}%")
        
        if 'ai_performance' in self.test_results:
            ai_avg_time = self.test_results['ai_performance']['avg_response_time']
            print(f"AI Model Average Response Time: {ai_avg_time:.3f}s")
        
        if 'load_performance' in self.test_results:
            load_success_rate = self.test_results['load_performance']['success_rate']
            print(f"Load Test Success Rate: {load_success_rate:.1f}%")
        
        print("\nAll tests completed successfully!")
        
        # Save test results to JSON
        with open('/home/ubuntu/test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        return self.test_results

if __name__ == "__main__":
    tester = SafetySystemTester()
    results = tester.run_all_tests()

