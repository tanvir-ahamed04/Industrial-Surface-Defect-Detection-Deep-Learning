# performance_benchmark.py
import torch
import time
import psutil
import GPUtil
from tabulate import tabulate

class PerformanceBenchmark:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        from retinanet_train import get_retinanet
        model = get_retinanet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def benchmark_inference(self, num_runs=100, image_size=(512, 512)):
        """Benchmark inference speed"""
        print(f"\nBenchmarking inference speed ({num_runs} runs)...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *image_size).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)
        
        # Benchmark
        times = []
        memory_usage = []
        
        for i in range(num_runs):
            # Measure memory before
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()
            else:
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Inference
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Measure memory after
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_used = (memory_after - memory_before) / 1024 / 1024
            else:
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = memory_after - memory_before
            
            times.append(end_time - start_time)
            memory_usage.append(memory_used)
        
        avg_time = sum(times) / len(times)
        avg_memory = sum(memory_usage) / len(memory_usage)
        fps = 1 / avg_time
        
        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'avg_memory_usage': avg_memory,
            'device': str(self.device)
        }
    
    def benchmark_batch(self, batch_sizes=[1, 2, 4, 8], image_size=(512, 512)):
        """Benchmark different batch sizes"""
        print(f"\nBenchmarking different batch sizes...")
        
        results = []
        for batch_size in batch_sizes:
            # Create dummy batch
            dummy_batch = torch.randn(batch_size, 3, *image_size).to(self.device)
            
            # Warmup
            _ = self.model(dummy_batch)
            
            # Benchmark
            times = []
            for _ in range(50):
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model(dummy_batch)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            fps = batch_size / avg_time
            
            results.append({
                'batch_size': batch_size,
                'avg_time': avg_time,
                'fps': fps
            })
        
        return results
    
    def get_system_info(self):
        """Get system information"""
        info = {}
        
        # CPU
        info['cpu_cores'] = psutil.cpu_count(logical=True)
        info['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'
        
        # RAM
        ram = psutil.virtual_memory()
        info['ram_total'] = f"{ram.total / 1024**3:.1f} GB"
        info['ram_available'] = f"{ram.available / 1024**3:.1f} GB"
        
        # GPU
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info['gpu_name'] = gpu.name
                info['gpu_memory'] = f"{gpu.memoryTotal} MB"
                info['gpu_utilization'] = f"{gpu.load * 100:.1f}%"
        else:
            info['gpu_name'] = 'None'
        
        return info
    
    def run_full_benchmark(self):
        """Run complete benchmark"""
        print("=" * 60)
        print("PERFORMANCE BENCHMARK FOR DEFECT DETECTION SYSTEM")
        print("=" * 60)
        
        # System info
        print("\n[1] SYSTEM INFORMATION")
        system_info = self.get_system_info()
        for key, value in system_info.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Inference benchmark
        print("\n[2] INFERENCE PERFORMANCE")
        inference_results = self.benchmark_inference()
        for key, value in inference_results.items():
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}" if isinstance(value, float) else f"  {key.replace('_', ' ').title()}: {value}")
        
        # Batch benchmark
        print("\n[3] BATCH PERFORMANCE")
        batch_results = self.benchmark_batch()
        print(tabulate(batch_results, headers='keys', tablefmt='grid'))
        
        # Summary
        print("\n[4] SUMMARY")
        print("✓ Ready for real-time application" if inference_results['fps'] > 30 else 
              "⚠ May need optimization for real-time")
        print("✓ GPU acceleration enabled" if torch.cuda.is_available() else 
              "⚠ Running on CPU - consider using GPU for faster inference")

if __name__ == "__main__":
    benchmark = PerformanceBenchmark('best_retinanet.pth')
    benchmark.run_full_benchmark()