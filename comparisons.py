from termcolor import colored
class comparison:
    def __init__(self):
        self.comparisons = []

    def add_comparison(self, comparison_data):
        self.comparisons.append(comparison_data)

    def display_comparisons(self):
        print("\nHeuristic Comparisons:")
        print("-" * 50)
        for comp in self.comparisons:
            print(colored(f"\nHeuristic: {comp['heuristic_name']}", "cyan"))
            print(f"Time Taken: {comp['time_taken']:.2f} seconds")
            print(f"Steps: {comp['steps']}")
            print(f"Heuristic Value: {comp['heuristic_value']:.2f}")
            solution = comp['solution'] if comp['solution'] else "No solution"
            print(f"Solution: {solution}")
            print("Nodes Evaluated: ", comp['nodes_evaluated'])
            print("-" * 50)

if __name__ == "__main__":
    comp = comparison()
    comp.display_comparisons()