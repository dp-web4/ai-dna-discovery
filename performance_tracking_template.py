#!/usr/bin/env python3
"""
Performance Tracking System Template
Copy and customize this for your specific experiment domain

CUSTOMIZATION CHECKLIST:
1. Update the table schema for your metrics
2. Modify test_type options 
3. Add domain-specific additional_metrics examples
4. Update the initialization examples
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
        # CUSTOMIZE: Add your domain-specific fields here
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                test_type TEXT NOT NULL,
                who_ran TEXT NOT NULL,
                success BOOLEAN DEFAULT TRUE,
                
                -- CUSTOMIZE: Add your primary metrics here
                -- Examples for different domains:
                
                -- Vision/Graphics metrics
                -- fps_avg REAL,
                -- fps_min REAL,
                -- fps_max REAL,
                -- processing_time_ms REAL,
                -- resolution TEXT,
                -- gpu_used BOOLEAN,
                -- gpu_library TEXT,
                
                -- LLM/AI metrics  
                -- model_name TEXT,
                -- tokens_per_second REAL,
                -- memory_usage_mb REAL,
                -- accuracy REAL,
                -- latency_ms REAL,
                
                -- Hardware/Battery metrics
                -- voltage REAL,
                -- current REAL,
                -- power REAL,
                -- efficiency_percent REAL,
                -- temperature_c REAL,
                
                notes TEXT,
                error_message TEXT,
                system_info TEXT
            )
        """)
        
        # Additional metrics table for flexibility
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
        
    def record_test(self, 
                   filename: str, 
                   test_type: str, 
                   who_ran: str,
                   success: bool = True,
                   notes: Optional[str] = None,
                   error_message: Optional[str] = None,
                   additional_metrics: Optional[Dict[str, tuple]] = None,
                   **kwargs):
        """
        Record a test run with flexible metrics
        
        Args:
            filename: Script/notebook that was tested
            test_type: Type of test (customize for your domain)
            who_ran: Who ran the test (user, claude, automated)
            success: Whether the test completed successfully
            notes: Additional context
            error_message: Error details if success=False
            additional_metrics: Dict of {metric_name: (value, unit)}
            **kwargs: Any additional fields defined in your schema
        """
        # CUSTOMIZE: Update system info for your domain
        system_info = {
            "platform": os.uname().sysname,
            "hostname": os.uname().nodename,
            # Add domain-specific system info
        }
        
        # Build the INSERT query dynamically based on kwargs
        fields = ["filename", "test_type", "who_ran", "success", "notes", 
                 "error_message", "system_info"]
        values = [filename, test_type, who_ran, success, notes, 
                 error_message, json.dumps(system_info)]
        
        # Add any additional fields from kwargs
        for key, value in kwargs.items():
            fields.append(key)
            values.append(value)
        
        placeholders = ",".join(["?" for _ in values])
        fields_str = ",".join(fields)
        
        cursor = self.conn.execute(
            f"INSERT INTO test_runs ({fields_str}) VALUES ({placeholders})",
            values
        )
        
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
        
    def search_tests(self, 
                    filename: Optional[str] = None,
                    test_type: Optional[str] = None,
                    success_only: bool = False,
                    who_ran: Optional[str] = None,
                    days_back: int = 30,
                    **kwargs) -> List[Dict]:
        """Search for test results with flexible criteria"""
        query = "SELECT * FROM test_runs WHERE 1=1"
        params = []
        
        if filename:
            query += " AND filename LIKE ?"
            params.append(f"%{filename}%")
            
        if test_type:
            query += " AND test_type = ?"
            params.append(test_type)
            
        if success_only:
            query += " AND success = 1"
            
        if who_ran:
            query += " AND who_ran = ?"
            params.append(who_ran)
            
        # Add any custom field searches from kwargs
        for key, value in kwargs.items():
            if value is not None:
                query += f" AND {key} = ?"
                params.append(value)
            
        query += " AND timestamp > datetime('now', '-{} days')".format(days_back)
        query += " ORDER BY timestamp DESC"
        
        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
        
    def get_test_details(self, test_run_id: int) -> Dict:
        """Get full details of a test run including additional metrics"""
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
        
        if not tests:
            print("No recent tests found.")
            return
            
        print(f"\n{'='*100}")
        print(f"{'Timestamp':<20} {'File':<30} {'Type':<15} {'Success':<8} {'Who':<10}")
        print(f"{'='*100}")
        
        for test in tests:
            timestamp = test['timestamp'][:19]
            filename = os.path.basename(test['filename'])
            if len(filename) > 30:
                filename = filename[:27] + "..."
            test_type = test['test_type']
            success = "✓" if test['success'] else "✗"
            who = test['who_ran']
            
            print(f"{timestamp:<20} {filename:<30} {test_type:<15} {success:<8} {who:<10}")
            
            if test['notes']:
                print(f"  Notes: {test['notes']}")
            if not test['success'] and test['error_message']:
                print(f"  Error: {test['error_message']}")
                
        print(f"{'='*100}\n")

def main():
    """Initialize the database with example entries"""
    tracker = PerformanceTracker()
    
    # CUSTOMIZE: Add initialization examples for your domain
    print("Initializing performance tracking database...")
    print("Database created: performance_tests.db")
    print("\nCustomize this template for your specific domain:")
    print("1. Update the table schema in create_tables()")
    print("2. Modify test_type options for your domain")
    print("3. Add relevant metrics and fields")
    print("4. Update the examples below")
    
    # Example: Vision domain
    # tracker.record_test(
    #     filename="object_detection.py",
    #     test_type="realtime",
    #     who_ran="user",
    #     fps_avg=30.5,
    #     gpu_used=True,
    #     notes="YOLO v5 on street scenes"
    # )
    
    # Example: LLM domain
    # tracker.record_test(
    #     filename="llm_inference.py",
    #     test_type="benchmark",
    #     who_ran="automated",
    #     model_name="llama-7b",
    #     tokens_per_second=42.3,
    #     memory_usage_mb=4096,
    #     notes="Batch size 1, 8-bit quantization"
    # )
    
    tracker.print_summary()

if __name__ == "__main__":
    main()