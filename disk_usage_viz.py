"""
Disk usage visualizer (CLI + matplotlib)
Usage examples:
    python3 disk_usage_viz.py /Users/you/Downloads
    python3 disk_usage_viz.py / -n 15 --save image.png
"""

import os
import argparse
import matplotlib.pyplot as plt

def get_dir_sizes(path):
    """Walk `path` and return a dict mapping relative_dir -> total_size_bytes."""
    dir_sizes = {}
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                full_path = os.path.join(root, name)

                # Windows long path handling:
                if os.name == "nt":
                    full_path = r"\\?\\" + os.path.abspath(full_path)

                size = os.path.getsize(full_path)
                dir_path = os.path.relpath(root, path)
                dir_sizes[dir_path] = dir_sizes.get(dir_path, 0) + size
            except (FileNotFoundError, PermissionError, OSError):
                # Skip files we can't access for any reason
                continue
    return dir_sizes

def pretty_print_table(sorted_dirs, top_n):
    """Prints a simple table of top N directories with sizes in MB."""
    print(f"{'Directory':40s} | {'Size (MB)':>10s}")
    print("-" * 60)
    for dir_path, size in sorted_dirs[:top_n]:
        size_mb = size / (1024 * 1024)
        print(f"{dir_path:40s} | {size_mb:10.2f}")
    print("-" * 60)

def plot_bar_chart(sorted_dirs, top_n, title="Top directories by size", save_path=None, show=True):
    """Plot a bar chart (matplotlib). Requirements: single plot, matplotlib, no explicit colors."""
    top = sorted_dirs[:top_n]
    if not top:
        print("No directories to plot.")
        return

    labels = [d for d, s in top]
    sizes_mb = [s / (1024 * 1024) for d, s in top]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, sizes_mb)          # do not explicitly set colors
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Size (MB)")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Disk Usage Visualizer (CLI + plot)")
    parser.add_argument("path", nargs="?", default=".", help="Path to analyze")
    parser.add_argument("-n", "--top", type=int, default=10, help="Show top N largest directories")
    parser.add_argument("--save", help="Save bar chart as PNG file (optional)")
    parser.add_argument("--no-show", action="store_true", help="Do not show the interactive plot (useful for headless)")
    args = parser.parse_args()

    print(f"Analyzing disk usage in: {args.path}\n")
    dir_sizes = get_dir_sizes(args.path)

    # If the root directory itself contains files, relpath('.') produces '.'
    # You may want to replace '.' with the root name for prettier output.
    if "." in dir_sizes and args.path not in (".", "./"):
        dir_sizes[os.path.basename(os.path.abspath(args.path))] = dir_sizes.pop(".")

    sorted_dirs = sorted(dir_sizes.items(), key=lambda x: x[1], reverse=True)

    print(f"Top {args.top} largest directories:")
    pretty_print_table(sorted_dirs, args.top)

    # Plot and optionally save
    plot_bar_chart(sorted_dirs, args.top, title=f"Top {args.top} directories in {args.path}",
                   save_path=args.save, show=not args.no_show)

    print("✅ Disk usage analysis complete.")

if __name__ == "__main__":
    main()
