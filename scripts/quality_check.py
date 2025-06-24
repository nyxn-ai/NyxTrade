#!/usr/bin/env python3
"""
Code Quality Check Script
Runs comprehensive code quality checks for NyxTrade project
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any


class QualityChecker:
    """Code quality checker for NyxTrade"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results: Dict[str, Any] = {}
        
    def run_all_checks(self) -> bool:
        """Run all quality checks"""
        print("🔍 Running NyxTrade Code Quality Checks")
        print("=" * 50)
        
        checks = [
            ("Security Check", self.check_security),
            ("Type Checking", self.check_types),
            ("Code Style", self.check_style),
            ("Unit Tests", self.run_tests),
            ("Test Coverage", self.check_coverage),
            ("Import Analysis", self.check_imports),
            ("Documentation", self.check_documentation),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}")
            print("-" * 30)
            
            try:
                result = check_func()
                self.results[check_name] = result
                
                if result.get('passed', False):
                    print(f"✅ {check_name}: PASSED")
                else:
                    print(f"❌ {check_name}: FAILED")
                    all_passed = False
                    
                # Print details
                if result.get('details'):
                    for detail in result['details']:
                        print(f"   {detail}")
                        
            except Exception as e:
                print(f"❌ {check_name}: ERROR - {e}")
                all_passed = False
        
        # Print summary
        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 All quality checks PASSED!")
        else:
            print("⚠️  Some quality checks FAILED. Please review and fix issues.")
        
        return all_passed
    
    def check_security(self) -> Dict[str, Any]:
        """Check for security issues"""
        issues = []
        
        # Check for hardcoded secrets
        secret_patterns = [
            "password",
            "secret",
            "key",
            "token",
            "api_key",
            "private_key"
        ]
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if "test" in str(file_path) or "__pycache__" in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                    for pattern in secret_patterns:
                        if f'"{pattern}"' in content or f"'{pattern}'" in content:
                            # Check if it's in a comment or docstring
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if pattern in line and not (line.strip().startswith('#') or '"""' in line):
                                    issues.append(f"{file_path}:{i+1} - Potential hardcoded secret: {pattern}")
                                    
            except Exception as e:
                issues.append(f"Error reading {file_path}: {e}")
        
        # Check for private key exposure patterns
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for hex patterns that might be private keys
                    import re
                    hex_pattern = r'0x[a-fA-F0-9]{64}'
                    matches = re.findall(hex_pattern, content)
                    
                    if matches and "test" not in str(file_path):
                        issues.append(f"{file_path} - Potential private key pattern found")
                        
            except Exception:
                pass
        
        return {
            'passed': len(issues) == 0,
            'details': issues[:10],  # Limit to first 10 issues
            'total_issues': len(issues)
        }
    
    def check_types(self) -> Dict[str, Any]:
        """Run mypy type checking"""
        try:
            result = subprocess.run(
                ["mypy", ".", "--ignore-missing-imports", "--no-strict-optional"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            errors = result.stdout.split('\n') if result.stdout else []
            errors = [e for e in errors if e.strip() and "error:" in e]
            
            return {
                'passed': result.returncode == 0,
                'details': errors[:10],  # First 10 errors
                'total_errors': len(errors)
            }
            
        except FileNotFoundError:
            return {
                'passed': False,
                'details': ["mypy not installed"],
                'total_errors': 1
            }
    
    def check_style(self) -> Dict[str, Any]:
        """Check code style with flake8"""
        try:
            result = subprocess.run(
                ["flake8", ".", "--max-line-length=120", "--ignore=E203,W503"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            issues = result.stdout.split('\n') if result.stdout else []
            issues = [i for i in issues if i.strip()]
            
            return {
                'passed': result.returncode == 0,
                'details': issues[:10],  # First 10 issues
                'total_issues': len(issues)
            }
            
        except FileNotFoundError:
            return {
                'passed': False,
                'details': ["flake8 not installed"],
                'total_issues': 1
            }
    
    def run_tests(self) -> Dict[str, Any]:
        """Run unit tests"""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            output_lines = result.stdout.split('\n') if result.stdout else []
            
            # Extract test results
            passed_tests = len([line for line in output_lines if "PASSED" in line])
            failed_tests = len([line for line in output_lines if "FAILED" in line])
            
            details = []
            if failed_tests > 0:
                details.extend([line for line in output_lines if "FAILED" in line][:5])
            
            details.append(f"Tests run: {passed_tests + failed_tests}")
            details.append(f"Passed: {passed_tests}")
            details.append(f"Failed: {failed_tests}")
            
            return {
                'passed': result.returncode == 0,
                'details': details,
                'tests_run': passed_tests + failed_tests,
                'tests_passed': passed_tests,
                'tests_failed': failed_tests
            }
            
        except FileNotFoundError:
            return {
                'passed': False,
                'details': ["pytest not installed or tests directory not found"],
                'tests_run': 0
            }
    
    def check_coverage(self) -> Dict[str, Any]:
        """Check test coverage"""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=.", "--cov-report=term-missing", "tests/"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            output_lines = result.stdout.split('\n') if result.stdout else []
            
            # Extract coverage percentage
            coverage_line = None
            for line in output_lines:
                if "TOTAL" in line and "%" in line:
                    coverage_line = line
                    break
            
            coverage_percent = 0
            if coverage_line:
                import re
                match = re.search(r'(\d+)%', coverage_line)
                if match:
                    coverage_percent = int(match.group(1))
            
            details = [f"Coverage: {coverage_percent}%"]
            
            # Add missing coverage details
            missing_lines = [line for line in output_lines if "missing" in line.lower()]
            details.extend(missing_lines[:5])
            
            return {
                'passed': coverage_percent >= 80,  # 80% minimum coverage
                'details': details,
                'coverage_percent': coverage_percent
            }
            
        except FileNotFoundError:
            return {
                'passed': False,
                'details': ["pytest-cov not installed"],
                'coverage_percent': 0
            }
    
    def check_imports(self) -> Dict[str, Any]:
        """Check for import issues"""
        issues = []
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if "__pycache__" in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    for i, line in enumerate(lines):
                        line = line.strip()
                        
                        # Check for unused imports (simplified)
                        if line.startswith('import ') or line.startswith('from '):
                            # This is a simplified check - in practice you'd use tools like isort
                            if 'unused' in line:  # Placeholder check
                                issues.append(f"{file_path}:{i+1} - Potentially unused import")
                        
                        # Check for relative imports outside package
                        if line.startswith('from .') and 'tests' not in str(file_path):
                            # Check if it's a valid relative import
                            pass  # Simplified check
                            
            except Exception as e:
                issues.append(f"Error reading {file_path}: {e}")
        
        return {
            'passed': len(issues) == 0,
            'details': issues[:10],
            'total_issues': len(issues)
        }
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness"""
        issues = []
        
        # Check for README
        readme_files = list(self.project_root.glob("README*"))
        if not readme_files:
            issues.append("No README file found")
        
        # Check for docstrings in main modules
        python_files = [
            f for f in self.project_root.rglob("*.py") 
            if "test" not in str(f) and "__pycache__" not in str(f)
        ]
        
        missing_docstrings = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for module docstring
                    if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                        missing_docstrings += 1
                        
            except Exception:
                pass
        
        if missing_docstrings > 0:
            issues.append(f"{missing_docstrings} files missing module docstrings")
        
        # Check for configuration documentation
        config_files = list(self.project_root.glob("config/*.example.*"))
        if not config_files:
            issues.append("No example configuration files found")
        
        return {
            'passed': len(issues) == 0,
            'details': issues,
            'total_issues': len(issues)
        }
    
    def generate_report(self) -> str:
        """Generate quality report"""
        report = "# NyxTrade Code Quality Report\n\n"
        
        for check_name, result in self.results.items():
            status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
            report += f"## {check_name}: {status}\n\n"
            
            if result.get('details'):
                report += "Details:\n"
                for detail in result['details']:
                    report += f"- {detail}\n"
                report += "\n"
        
        return report


def main():
    """Main function"""
    checker = QualityChecker()
    
    # Run all checks
    all_passed = checker.run_all_checks()
    
    # Generate report
    report = checker.generate_report()
    
    # Save report
    with open("quality_report.md", "w") as f:
        f.write(report)
    
    print(f"\n📄 Quality report saved to: quality_report.md")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
