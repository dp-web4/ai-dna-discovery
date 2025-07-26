#!/usr/bin/env python3
"""
Performance Test Tracking System
Maintains accurate records of all vision experiment performance tests
"""

import sqlite3
import datetime
import os
import json
from typing import Optional, List, Dict

class PerformanceTracker:
    def __init__(self, db_path="performance_tests.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()
        
    def create_tables(self):
        """Create the performance tracking tables"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                test_type TEXT NOT NULL,
                fps_avg REAL,
                fps_min REAL,
                fps_max REAL,
                processing_time_ms REAL,
                gpu_used BOOLEAN,
                gpu_library TEXT,
                resolution TEXT,
                notes TEXT,
                system_info TEXT,
                who_ran TEXT NOT NULL
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS test_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_run_id INTEGER,
                metric_name TEXT,
                metric_value REAL,
                unit TEXT,
                FOREIGN KEY (test_run_id) REFERENCES test_runs (id)
            )
        """)
        
        self.conn.commit()
        
    def record_test(self, filename: str, test_type: str, who_ran: str,
                   fps_avg: Optional[float] = None,
                   fps_min: Optional[float] = None, 
                   fps_max: Optional[float] = None,
                   processing_time_ms: Optional[float] = None,
                   gpu_used: bool = False,
                   gpu_library: Optional[str] = None,
                   resolution: str = "1280x720",
                   notes: Optional[str] = None,
                   additional_metrics: Optional[Dict[str, tuple]] = None):
        """
        Record a performance test run
        
        Args:
            filename: Script that was run
            test_type: Type of test (realtime, benchmark, etc)
            who_ran: Who ran the test (user, claude, automated)
            additional_metrics: Dict of {metric_name: (value, unit)}
        """
        system_info = {
            "platform": "Jetson Orin Nano",
            "cuda_cores": 1024,
            "memory": "8GB LPDDR5",
            "jetpack": "6.2.1"
        }
        
        cursor = self.conn.execute("""
            INSERT INTO test_runs 
            (filename, test_type, who_ran, fps_avg, fps_min, fps_max, 
             processing_time_ms, gpu_used, gpu_library, resolution, notes, system_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (filename, test_type, who_ran, fps_avg, fps_min, fps_max,
              processing_time_ms, gpu_used, gpu_library, resolution, notes,
              json.dumps(system_info)))
        
        test_run_id = cursor.lastrowid
        
        # Record additional metrics
        if additional_metrics:
            for metric_name, (value, unit) in additional_metrics.items():
                self.conn.execute("""
                    INSERT INTO test_details (test_run_id, metric_name, metric_value, unit)
                    VALUES (?, ?, ?, ?)
                """, (test_run_id, metric_name, value, unit))
        
        self.conn.commit()
        return test_run_id
        
    def search_tests(self, filename: Optional[str] = None,
                    test_type: Optional[str] = None,
                    gpu_used: Optional[bool] = None,
                    min_fps: Optional[float] = None,
                    days_back: int = 30) -> List[Dict]:
        """Search for test results"""
        query = "SELECT * FROM test_runs WHERE 1=1"
        params = []
        
        if filename:
            query += " AND filename LIKE ?"
            params.append(f"%{filename}%")
            
        if test_type:
            query += " AND test_type = ?"
            params.append(test_type)
            
        if gpu_used is not None:
            query += " AND gpu_used = ?"
            params.append(gpu_used)
            
        if min_fps:
            query += " AND fps_avg >= ?"
            params.append(min_fps)
            
        query += " AND timestamp > datetime('now', '-{} days')".format(days_back)
        query += " ORDER BY timestamp DESC"
        
        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
        
    def get_test_details(self, test_run_id: int) -> Dict:
        """Get full details of a test run"""
        run = self.conn.execute(
            "SELECT * FROM test_runs WHERE id = ?", (test_run_id,)
        ).fetchone()
        
        if not run:
            return None
            
        details = self.conn.execute(
            "SELECT * FROM test_details WHERE test_run_id = ?", (test_run_id,)
        ).fetchall()
        
        result = dict(run)
        result['additional_metrics'] = [dict(d) for d in details]
        return result
        
    def print_summary(self, limit: int = 10):
        """Print a summary of recent tests"""
        tests = self.search_tests(days_back=7)[:limit]
        
        print(f"\n{'='*100}")
        print(f"{'Timestamp':<20} {'File':<35} {'Type':<10} {'FPS':<10} {'GPU':<5} {'Who':<10}")
        print(f"{'='*100}")
        
        for test in tests:
            timestamp = test['timestamp'][:19]
            filename = os.path.basename(test['filename'])
            if len(filename) > 35:
                filename = filename[:32] + "..."
            test_type = test['test_type']
            fps = f"{test['fps_avg']:.1f}" if test['fps_avg'] else "N/A"
            gpu = "Yes" if test['gpu_used'] else "No"
            who = test['who_ran']
            
            print(f"{timestamp:<20} {filename:<35} {test_type:<10} {fps:<10} {gpu:<5} {who:<10}")
            
            if test['notes']:
                print(f"  Notes: {test['notes']}")
        print(f"{'='*100}\n")

def main():
    """Initialize the database with known test results"""
    tracker = PerformanceTracker()
    
    # Record the actual test results we know about
    print("Recording known test results...")
    
    # 1. The benchmark test Claude ran
    tracker.record_test(
        filename="performance_benchmark.py",
        test_type="benchmark",
        who_ran="claude",
        fps_avg=118.9,
        processing_time_ms=8.41,
        gpu_used=False,
        notes="CPU benchmark showing theoretical max FPS",
        additional_metrics={
            "color_conversion_ms": (1.94, "ms"),
            "absdiff_ms": (0.22, "ms"),
            "gaussian_blur_ms": (3.56, "ms"),
            "threshold_ms": (0.10, "ms"),
            "motion_grid_ms": (2.58, "ms")
        }
    )
    
    # 2. The GPU test that was slow
    tracker.record_test(
        filename="consciousness_attention_gpu.py",
        test_type="realtime",
        who_ran="user",
        fps_avg=3.0,
        gpu_used=True,
        gpu_library="cupy",
        notes="Poor performance due to memory transfer overhead"
    )
    
    # 3. The optimized CPU version
    tracker.record_test(
        filename="consciousness_attention_minimal_gpu.py",
        test_type="realtime",
        who_ran="user",
        fps_avg=30.0,
        gpu_used=False,
        notes="Camera-limited to 30 FPS, smooth real-time performance"
    )
    
    # 4. The clean consciousness version
    tracker.record_test(
        filename="consciousness_attention_clean.py",
        test_type="realtime",
        who_ran="user",
        fps_avg=25.0,
        fps_min=20.0,
        fps_max=25.0,
        gpu_used=False,
        notes="Best biological model, good motion response"
    )
    
    # Print summary
    tracker.print_summary()
    
    print("\nDatabase created: performance_tests.db")
    print("Use search_performance.py to query results")

if __name__ == "__main__":
    main()