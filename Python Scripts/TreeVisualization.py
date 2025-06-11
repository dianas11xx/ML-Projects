import re

def parse_weka_j48_tree(tree_text):
    """
    Parses a Weka J48 text-based tree output into a list of nodes.

    Args:
        tree_text (str): The multi-line string output from Weka J48.

    Returns:
        list: A list of dictionaries, where each dictionary represents a node
              and contains its level, text, and potentially its prediction.
    """
    nodes = []
    lines = tree_text.strip().split('\n')

    for line in lines:
        if not line.strip():
            continue

        # Calculate indentation level (each '|   ' is one level)
        level = 0
        while line.startswith('|   '):
            level += 1
            line = line[4:] # Remove '|   '

        node_text = line.strip()
        prediction = None
        instances = None
        error = None # Initialize error to None

        # Check if it's a leaf node (ends with ': PREDICTION (count/error)' or ': PREDICTION (count)')
        # This regex now allows for both formats: (count/error) or just (count)
        leaf_match = re.match(r'(.*):\s*(POS|NEG)\s*\((\d+\.?\d*)(?:/(\d+\.?\d*))?\)$', node_text)
        if leaf_match:
            node_text = leaf_match.group(1).strip()
            prediction = leaf_match.group(2)
            instances = float(leaf_match.group(3))
            if leaf_match.group(4): # Check if the error part was captured
                error = float(leaf_match.group(4))
            else:
                error = 0.0 # If no error is specified, assume it's 0.0 (pure leaf)

        nodes.append({
            'level': level,
            'text': node_text,
            'prediction': prediction,
            'instances': instances,
            'error': error, # Always include error, even if None or 0.0
            'parent_node_id': None, # To be filled during tree construction
            'parent_relation': None, # To be filled during tree construction
            'node_id': None # To be filled later
        })
    return nodes

def build_tree_structure(parsed_nodes):
    """
    Builds a hierarchical tree structure from parsed nodes and assigns unique IDs.

    Args:
        parsed_nodes (list): List of node dictionaries from parse_weka_j48_tree.

    Returns:
        tuple: (root_node_id, nodes_dict), where root_node_id is the ID of the root node
               and nodes_dict is a dictionary mapping node IDs to their data.
    """
    nodes_dict = {} # Maps node_id to node data
    node_counter = 0
    stack = [] # Used to keep track of potential parent nodes

    root_node_id = None

    for i, node_data in enumerate(parsed_nodes):
        node_id = f"N{node_counter}"
        node_data['node_id'] = node_id
        nodes_dict[node_id] = node_data
        node_counter += 1

        if node_data['level'] == 0:
            # This is the root node
            root_node_id = node_id
            stack = [(node_id, 0)] # (node_id, level)
        else:
            # Pop from stack until we find a parent at the correct level
            while stack and stack[-1][1] >= node_data['level']:
                stack.pop()

            if stack:
                parent_id, parent_level = stack[-1]
                node_data['parent_node_id'] = parent_id
                node_data['parent_relation'] = node_data['text'] # The condition that led to this node

            stack.append((node_id, node_data['level']))
    return root_node_id, nodes_dict


def generate_dot_from_tree(root_id, nodes_dict):
    """
    Generates a Graphviz DOT string from the parsed tree structure.

    Args:
        root_id (str): The ID of the root node.
        nodes_dict (dict): Dictionary mapping node IDs to their data.

    Returns:
        str: The Graphviz DOT string.
    """
    dot_string = "digraph J48Tree {\n"
    dot_string += "  graph [rankdir=TB];\n" # Top to Bottom layout
    dot_string += "  node [shape=box, style=filled, fillcolor=lightgray];\n"
    dot_string += "  edge [fontsize=10];\n" # Smaller font for edge labels

    # Define nodes
    for node_id, data in nodes_dict.items():
        node_label = ""

        # Determine node label (what's *inside* the box)
        if data['prediction']:
            # Leaf node
            # Format depends on whether an error value exists
            if data['error'] is not None:
                node_label = f"{data['prediction']} ({data['instances']}/{data['error']:.2f})"
            else: # Should not happen with the updated parse_weka_j48_tree, but good for robustness
                node_label = f"{data['prediction']} ({data['instances']})"
            dot_string += f"  {node_id} [label=\"{node_label}\", fillcolor=lightblue];\n"
        else:
            # Internal node: extract the feature name for the node label
            # Use data['text'] which holds the full condition for the edge
            match = re.match(r'([a-zA-Z0-9_.-]+)\s*[<=>!]+\s*(\d+\.?\d*)$', data['text'])
            if match:
                feature_name = match.group(1)
                node_label = feature_name # The actual feature being split on
            else:
                node_label = data['text'] # Fallback if regex doesn't match a clear feature

            dot_string += f"  {node_id} [label=\"{node_label}\"];\n"

    # Define edges
    for node_id, data in nodes_dict.items():
        if data['parent_node_id']:
            # The edge label is the condition that leads to this node
            edge_label = data['parent_relation']
            dot_string += f"  {data['parent_node_id']} -> {node_id} [label=\"{edge_label}\"];\n"

    dot_string += "}\n"
    return dot_string

weka_output = """
CHEMBL312_non_agonist_Label-2 <= 0
|   CHEMBL614805_non_agonist_Label-2 <= 3
|   |   CHEMBL2366069_non_agonist_Label-2 <= 2
|   |   |   CHEMBL382_non_agonist_Label-2 <= 2
|   |   |   |   CHEMBL394_non_agonist_Label-2 <= 1
|   |   |   |   |   CHEMBL2056_non_agonist_Label-2 <= 1
|   |   |   |   |   |   CHEMBL231_non_agonist_Label-2 <= 1: NEG (30.07/14.97)
|   |   |   |   |   |   CHEMBL231_non_agonist_Label-2 > 1: NEG (17.29/7.08)
|   |   |   |   |   CHEMBL2056_non_agonist_Label-2 > 1
|   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2: NEG (14.83/7.08)
|   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2: POS (15.7/6.18)
|   |   |   |   CHEMBL394_non_agonist_Label-2 > 1
|   |   |   |   |   CHEMBL288_non_agonist_Label-2 <= 1
|   |   |   |   |   |   CHEMBL233_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL2056_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   CHEMBL217_non_agonist_Label-2 <= 1: NEG (19.28/8.71)
|   |   |   |   |   |   |   |   CHEMBL217_non_agonist_Label-2 > 1: NEG (15.54/5.38)
|   |   |   |   |   |   |   CHEMBL2056_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   CHEMBL222_non_agonist_Label-2 <= 1: NEG (9.33/4.58)
|   |   |   |   |   |   |   |   CHEMBL222_non_agonist_Label-2 > 1: NEG (9.47/4.71)
|   |   |   |   |   |   CHEMBL233_non_agonist_Label-2 > 1: POS (16.14/6.81)
|   |   |   |   |   CHEMBL288_non_agonist_Label-2 > 1: NEG (14.29/4.01)
|   |   |   CHEMBL382_non_agonist_Label-2 > 2
|   |   |   |   CHEMBL288_non_agonist_Label-2 <= 1
|   |   |   |   |   CHEMBL364_non_agonist_Label-2 <= 2
|   |   |   |   |   |   CHEMBL2056_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL231_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2: NEG (25.95/12.88)
|   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2: NEG (14.58/6.92)
|   |   |   |   |   |   |   CHEMBL231_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   CHEMBL229_non_agonist_Label-2 <= 1: NEG (10.66/4.42)
|   |   |   |   |   |   |   |   CHEMBL229_non_agonist_Label-2 > 1: NEG (11.18/3.85)
|   |   |   |   |   |   CHEMBL2056_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2: NEG (19.52/9.18)
|   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2
|   |   |   |   |   |   |   |   CHEMBL240_non_agonist_Label-2 <= 1: POS (15.47/6.64)
|   |   |   |   |   |   |   |   CHEMBL240_non_agonist_Label-2 > 1: POS (10.8/3.37)
|   |   |   |   |   CHEMBL364_non_agonist_Label-2 > 2
|   |   |   |   |   |   CHEMBL4644_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2: NEG (12.42/5.73)
|   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2: POS (12.39/5.59)
|   |   |   |   |   |   CHEMBL4644_non_agonist_Label-2 > 1: POS (16.7/6.54)
|   |   |   |   CHEMBL288_non_agonist_Label-2 > 1: NEG (15.79/5.73)
|   |   CHEMBL2366069_non_agonist_Label-2 > 2
|   |   |   CHEMBL368_non_agonist_Label-2 <= 1
|   |   |   |   CHEMBL288_non_agonist_Label-2 <= 1
|   |   |   |   |   CHEMBL2056_non_agonist_Label-2 <= 1
|   |   |   |   |   |   CHEMBL256_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL229_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2: NEG (20.55/9.06)
|   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2: NEG (16.12/7.87)
|   |   |   |   |   |   |   CHEMBL229_non_agonist_Label-2 > 1: NEG (10.76/3.68)
|   |   |   |   |   |   CHEMBL256_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL222_non_agonist_Label-2 <= 1: POS (18.53/8.22)
|   |   |   |   |   |   |   CHEMBL222_non_agonist_Label-2 > 1: NEG (11.2/5.14)
|   |   |   |   |   CHEMBL2056_non_agonist_Label-2 > 1
|   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2
|   |   |   |   |   |   |   CHEMBL213_non_agonist_Label-2 <= 1: POS (10.59/5.08)
|   |   |   |   |   |   |   CHEMBL213_non_agonist_Label-2 > 1: NEG (10.35/4.38)
|   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2
|   |   |   |   |   |   |   CHEMBL1951_non_agonist_Label-2 <= 1: POS (11.78/5.47)
|   |   |   |   |   |   |   CHEMBL1951_non_agonist_Label-2 > 1: POS (17.47/5.53)
|   |   |   |   CHEMBL288_non_agonist_Label-2 > 1: NEG (13.5/5.0)
|   |   |   CHEMBL368_non_agonist_Label-2 > 1
|   |   |   |   CHEMBL364_non_agonist_Label-2 <= 2
|   |   |   |   |   CHEMBL2056_non_agonist_Label-2 <= 1
|   |   |   |   |   |   CHEMBL1867_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 <= 1: NEG (37.55/16.67)
|   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 > 1: POS (9.6/3.82)
|   |   |   |   |   |   CHEMBL1867_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL228_non_agonist_Label-2 <= 1: NEG (17.58/6.65)
|   |   |   |   |   |   |   CHEMBL228_non_agonist_Label-2 > 1: NEG (10.31/4.45)
|   |   |   |   |   CHEMBL2056_non_agonist_Label-2 > 1
|   |   |   |   |   |   CHEMBL2069_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2: NEG (16.46/7.97)
|   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2
|   |   |   |   |   |   |   |   CHEMBL264_non_agonist_Label-2 <= 1: POS (11.32/4.94)
|   |   |   |   |   |   |   |   CHEMBL264_non_agonist_Label-2 > 1: POS (15.21/4.67)
|   |   |   |   |   |   CHEMBL2069_non_agonist_Label-2 > 1: NEG (10.5/4.11)
|   |   |   |   CHEMBL364_non_agonist_Label-2 > 2
|   |   |   |   |   CHEMBL4303835_non_agonist_Label-2 <= 1
|   |   |   |   |   |   CHEMBL225_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL330_non_agonist_Label-2 <= 1: NEG (14.47/5.9)
|   |   |   |   |   |   |   CHEMBL330_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 <= 1: NEG (9.44/4.09)
|   |   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   CHEMBL224_agonist_Label-2 <= 1: POS (12.45/5.09)
|   |   |   |   |   |   |   |   |   CHEMBL224_agonist_Label-2 > 1: NEG (10.08/5.01)
|   |   |   |   |   |   CHEMBL225_non_agonist_Label-2 > 1: POS (18.15/7.48)
|   |   |   |   |   CHEMBL4303835_non_agonist_Label-2 > 1: POS (14.47/3.62)
|   CHEMBL614805_non_agonist_Label-2 > 3
|   |   CHEMBL1909044_non_agonist_Label-2 <= 1
|   |   |   CHEMBL319_non_agonist_Label-2 <= 1
|   |   |   |   CHEMBL249_non_agonist_Label-2 <= 1
|   |   |   |   |   CHEMBL208_agonist_Label-2 <= 0: POS (18.58/5.64)
|   |   |   |   |   CHEMBL208_agonist_Label-2 > 0
|   |   |   |   |   |   CHEMBL259_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 <= 1: NEG (14.17/7.06)
|   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 > 1: POS (9.58/3.74)
|   |   |   |   |   |   CHEMBL259_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL228_non_agonist_Label-2 <= 1: NEG (9.97/4.27)
|   |   |   |   |   |   |   CHEMBL228_non_agonist_Label-2 > 1: NEG (10.01/4.95)
|   |   |   |   CHEMBL249_non_agonist_Label-2 > 1
|   |   |   |   |   CHEMBL364_non_agonist_Label-2 <= 2
|   |   |   |   |   |   CHEMBL233_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 1: NEG (16.83/5.84)
|   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   CHEMBL1889_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 <= 1: NEG (19.99/9.58)
|   |   |   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 > 1: POS (9.73/3.99)
|   |   |   |   |   |   |   |   CHEMBL1889_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   CHEMBL228_non_agonist_Label-2 <= 1: NEG (9.17/3.83)
|   |   |   |   |   |   |   |   |   CHEMBL228_non_agonist_Label-2 > 1: NEG (11.06/5.26)
|   |   |   |   |   |   CHEMBL233_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL240_non_agonist_Label-2 <= 1: POS (12.18/5.66)
|   |   |   |   |   |   |   CHEMBL240_non_agonist_Label-2 > 1: POS (11.09/4.15)
|   |   |   |   |   CHEMBL364_non_agonist_Label-2 > 2
|   |   |   |   |   |   CHEMBL2056_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 <= 1: NEG (19.49/9.24)
|   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 > 1: POS (9.67/4.27)
|   |   |   |   |   |   CHEMBL2056_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL238_non_agonist_Label-2 <= 1: POS (17.93/5.84)
|   |   |   |   |   |   |   CHEMBL238_non_agonist_Label-2 > 1: POS (10.78/5.03)
|   |   |   CHEMBL319_non_agonist_Label-2 > 1
|   |   |   |   CHEMBL612544_non_agonist_Label-2 <= 2
|   |   |   |   |   CHEMBL288_non_agonist_Label-2 <= 1
|   |   |   |   |   |   CHEMBL2056_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL231_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2: NEG (26.36/12.57)
|   |   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2: NEG (16.57/7.32)
|   |   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 > 1: POS (11.97/4.32)
|   |   |   |   |   |   |   CHEMBL231_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   CHEMBL206_non_agonist_Label-2 <= 1: NEG (20.77/7.5)
|   |   |   |   |   |   |   |   CHEMBL206_non_agonist_Label-2 > 1: POS (9.33/4.5)
|   |   |   |   |   |   CHEMBL2056_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2
|   |   |   |   |   |   |   |   CHEMBL217_non_agonist_Label-2 <= 1: POS (10.2/4.78)
|   |   |   |   |   |   |   |   CHEMBL217_non_agonist_Label-2 > 1: NEG (14.15/5.83)
|   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2
|   |   |   |   |   |   |   |   CHEMBL264_non_agonist_Label-2 <= 1: NEG (13.57/6.74)
|   |   |   |   |   |   |   |   CHEMBL264_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   CHEMBL228_non_agonist_Label-2 <= 1: POS (16.92/3.69)
|   |   |   |   |   |   |   |   |   CHEMBL228_non_agonist_Label-2 > 1: POS (10.97/4.26)
|   |   |   |   |   CHEMBL288_non_agonist_Label-2 > 1: NEG (15.7/5.61)
|   |   |   |   CHEMBL612544_non_agonist_Label-2 > 2
|   |   |   |   |   CHEMBL364_non_agonist_Label-2 <= 3
|   |   |   |   |   |   CHEMBL1940_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL233_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 <= 1: NEG (11.2/3.53)
|   |   |   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 > 1: NEG (9.05/3.46)
|   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   CHEMBL251_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   |   |   CHEMBL237_non_agonist_Label-2 <= 1: NEG (16.52/6.55)
|   |   |   |   |   |   |   |   |   |   CHEMBL237_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   |   |   CHEMBL2203_non_agonist_Label-2 <= 1: POS (13.24/5.9)
|   |   |   |   |   |   |   |   |   |   |   CHEMBL2203_non_agonist_Label-2 > 1: NEG (14.28/6.1)
|   |   |   |   |   |   |   |   |   CHEMBL251_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 <= 1: POS (9.35/4.23)
|   |   |   |   |   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 > 1: NEG (9.39/4.65)
|   |   |   |   |   |   |   CHEMBL233_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   CHEMBL234_non_agonist_Label-2 <= 1: POS (21.15/7.79)
|   |   |   |   |   |   |   |   CHEMBL234_non_agonist_Label-2 > 1: NEG (12.83/5.73)
|   |   |   |   |   |   CHEMBL1940_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL2056_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   CHEMBL213_non_agonist_Label-2 <= 1: NEG (29.08/14.27)
|   |   |   |   |   |   |   |   CHEMBL213_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 <= 1: POS (9.92/4.37)
|   |   |   |   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 > 1: POS (10.85/3.14)
|   |   |   |   |   |   |   CHEMBL2056_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 <= 1: NEG (9.82/4.81)
|   |   |   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 > 1: NEG (16.71/6.53)
|   |   |   |   |   CHEMBL364_non_agonist_Label-2 > 3
|   |   |   |   |   |   CHEMBL1941_agonist_Label-2 <= 1: NEG (18.84/8.59)
|   |   |   |   |   |   CHEMBL1941_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2: NEG (14.32/6.77)
|   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2
|   |   |   |   |   |   |   |   CHEMBL236_non_agonist_Label-2 <= 1: POS (13.0/4.98)
|   |   |   |   |   |   |   |   CHEMBL236_non_agonist_Label-2 > 1: POS (17.29/4.31)
|   |   CHEMBL1909044_non_agonist_Label-2 > 1
|   |   |   CHEMBL613508_non_agonist_Label-2 <= 2
|   |   |   |   CHEMBL364_non_agonist_Label-2 <= 1
|   |   |   |   |   CHEMBL1833_non_agonist_Label-2 <= 1: POS (20.04/9.89)
|   |   |   |   |   CHEMBL1833_non_agonist_Label-2 > 1
|   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 <= 1: NEG (11.34/5.48)
|   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 > 1: NEG (15.25/5.69)
|   |   |   |   CHEMBL364_non_agonist_Label-2 > 1
|   |   |   |   |   CHEMBL211_non_agonist_Label-2 <= 1
|   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2: NEG (27.71/11.85)
|   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2: NEG (15.26/6.33)
|   |   |   |   |   CHEMBL211_non_agonist_Label-2 > 1
|   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2
|   |   |   |   |   |   |   CHEMBL4644_non_agonist_Label-2 <= 1: NEG (10.95/4.49)
|   |   |   |   |   |   |   CHEMBL4644_non_agonist_Label-2 > 1: NEG (10.91/5.43)
|   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2
|   |   |   |   |   |   |   CHEMBL1867_agonist_Label-2 <= 1: POS (14.54/3.92)
|   |   |   |   |   |   |   CHEMBL1867_agonist_Label-2 > 1: POS (17.21/7.55)
|   |   |   CHEMBL613508_non_agonist_Label-2 > 2
|   |   |   |   CHEMBL364_non_agonist_Label-2 <= 2
|   |   |   |   |   CHEMBL4303835_non_agonist_Label-2 <= 0: POS (11.5/3.86)
|   |   |   |   |   CHEMBL4303835_non_agonist_Label-2 > 0
|   |   |   |   |   |   CHEMBL219_non_agonist_Label-2 <= 2
|   |   |   |   |   |   |   CHEMBL233_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   CHEMBL3356_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   |   CHEMBL217_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   |   |   CHEMBL237_non_agonist_Label-2 <= 1: NEG (18.51/7.22)
|   |   |   |   |   |   |   |   |   |   CHEMBL237_non_agonist_Label-2 > 1: POS (17.27/7.62)
|   |   |   |   |   |   |   |   |   CHEMBL217_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2
|   |   |   |   |   |   |   |   |   |   |   CHEMBL228_non_agonist_Label-2 <= 1: NEG (9.23/2.75)
|   |   |   |   |   |   |   |   |   |   |   CHEMBL228_non_agonist_Label-2 > 1: NEG (10.79/4.47)
|   |   |   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2: NEG (14.04/6.09)
|   |   |   |   |   |   |   |   CHEMBL3356_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   CHEMBL1941_agonist_Label-2 <= 1: NEG (20.09/8.08)
|   |   |   |   |   |   |   |   |   CHEMBL1941_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   |   CHEMBL1889_non_agonist_Label-2 <= 1: POS (14.84/6.48)
|   |   |   |   |   |   |   |   |   |   CHEMBL1889_non_agonist_Label-2 > 1: NEG (15.71/6.91)
|   |   |   |   |   |   |   CHEMBL233_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 <= 1: POS (15.63/6.81)
|   |   |   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 > 1: NEG (10.87/4.77)
|   |   |   |   |   |   CHEMBL219_non_agonist_Label-2 > 2
|   |   |   |   |   |   |   CHEMBL1916_non_agonist_Label-2 <= 2
|   |   |   |   |   |   |   |   CHEMBL217_non_agonist_Label-2 <= 1: NEG (23.36/10.78)
|   |   |   |   |   |   |   |   CHEMBL217_non_agonist_Label-2 > 1: NEG (17.83/7.17)
|   |   |   |   |   |   |   CHEMBL1916_non_agonist_Label-2 > 2: POS (17.96/6.68)
|   |   |   |   CHEMBL364_non_agonist_Label-2 > 2
|   |   |   |   |   CHEMBL395_non_agonist_Label-2 <= 2
|   |   |   |   |   |   CHEMBL6020_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 <= 1: NEG (13.9/6.08)
|   |   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 > 1: POS (14.61/6.91)
|   |   |   |   |   |   CHEMBL6020_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL264_non_agonist_Label-2 <= 1: NEG (14.63/6.44)
|   |   |   |   |   |   |   CHEMBL264_non_agonist_Label-2 > 1: POS (12.66/4.02)
|   |   |   |   |   CHEMBL395_non_agonist_Label-2 > 2
|   |   |   |   |   |   CHEMBL3762_non_agonist_Label-2 <= 1: NEG (14.82/6.28)
|   |   |   |   |   |   CHEMBL3762_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL259_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2: POS (17.9/8.7)
|   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2
|   |   |   |   |   |   |   |   |   CHEMBL2056_non_agonist_Label-2 <= 1: POS (13.17/5.74)
|   |   |   |   |   |   |   |   |   CHEMBL2056_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   |   CHEMBL236_non_agonist_Label-2 <= 1: POS (10.97/3.2)
|   |   |   |   |   |   |   |   |   |   CHEMBL236_non_agonist_Label-2 > 1: POS (15.43/2.79)
|   |   |   |   |   |   |   CHEMBL259_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   CHEMBL206_non_agonist_Label-2 <= 1: NEG (8.92/3.83)
|   |   |   |   |   |   |   |   CHEMBL206_non_agonist_Label-2 > 1: POS (9.03/4.13)
CHEMBL312_non_agonist_Label-2 > 0
|   CHEMBL614925_non_agonist_Label-2 <= 2
|   |   CHEMBL288_non_agonist_Label-2 <= 1
|   |   |   CHEMBL2056_non_agonist_Label-2 <= 1
|   |   |   |   CHEMBL231_non_agonist_Label-2 <= 1
|   |   |   |   |   CHEMBL211_non_agonist_Label-2 <= 1
|   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2: NEG (28.26/13.28)
|   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2: NEG (16.45/7.31)
|   |   |   |   |   CHEMBL211_non_agonist_Label-2 > 1: POS (12.76/4.75)
|   |   |   |   CHEMBL231_non_agonist_Label-2 > 1
|   |   |   |   |   CHEMBL214_non_agonist_Label-2 <= 1
|   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 <= 1: NEG (15.0/6.57)
|   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 > 1: NEG (9.6/3.51)
|   |   |   |   |   CHEMBL214_non_agonist_Label-2 > 1: NEG (11.78/3.55)
|   |   |   CHEMBL2056_non_agonist_Label-2 > 1
|   |   |   |   CHEMBL230_non_agonist_Label-2 <= 1
|   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2
|   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 <= 1: POS (9.89/4.42)
|   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 > 1: NEG (10.51/4.49)
|   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2
|   |   |   |   |   |   CHEMBL264_non_agonist_Label-2 <= 1: POS (11.47/5.34)
|   |   |   |   |   |   CHEMBL264_non_agonist_Label-2 > 1: POS (16.91/4.8)
|   |   |   |   CHEMBL230_non_agonist_Label-2 > 1: NEG (11.02/4.31)
|   |   CHEMBL288_non_agonist_Label-2 > 1
|   |   |   CHEMBL211_non_agonist_Label-2 <= 1: NEG (9.58/3.75)
|   |   |   CHEMBL211_non_agonist_Label-2 > 1: NEG (10.14/2.89)
|   CHEMBL614925_non_agonist_Label-2 > 2
|   |   CHEMBL376_non_agonist_Label-2 <= 2
|   |   |   CHEMBL3879801_non_agonist_Label-2 <= 1
|   |   |   |   CHEMBL2056_agonist_Label-2 <= 1: NEG (17.45/8.02)
|   |   |   |   CHEMBL2056_agonist_Label-2 > 1
|   |   |   |   |   CHEMBL211_non_agonist_Label-2 <= 1: NEG (9.54/3.07)
|   |   |   |   |   CHEMBL211_non_agonist_Label-2 > 1: NEG (11.2/4.59)
|   |   |   CHEMBL3879801_non_agonist_Label-2 > 1
|   |   |   |   CHEMBL2056_non_agonist_Label-2 <= 1
|   |   |   |   |   CHEMBL216_non_agonist_Label-2 <= 1: NEG (27.4/12.92)
|   |   |   |   |   CHEMBL216_non_agonist_Label-2 > 1: NEG (9.8/3.44)
|   |   |   |   CHEMBL2056_non_agonist_Label-2 > 1
|   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 2: NEG (13.49/6.11)
|   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 2
|   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 <= 1: POS (11.31/3.66)
|   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 > 1: POS (10.7/4.28)
|   |   CHEMBL376_non_agonist_Label-2 > 2
|   |   |   CHEMBL364_non_agonist_Label-2 <= 2
|   |   |   |   CHEMBL3622_non_agonist_Label-2 <= 1
|   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 1
|   |   |   |   |   |   CHEMBL1867_non_agonist_Label-2 <= 1: NEG (16.93/4.81)
|   |   |   |   |   |   CHEMBL1867_non_agonist_Label-2 > 1: NEG (14.51/6.84)
|   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 1
|   |   |   |   |   |   CHEMBL1997_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   CHEMBL251_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   CHEMBL2056_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 <= 3: NEG (30.37/13.16)
|   |   |   |   |   |   |   |   |   CHEMBL612545_non_agonist_Label-2 > 3: POS (14.07/6.05)
|   |   |   |   |   |   |   |   CHEMBL2056_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   CHEMBL1941_agonist_Label-2 <= 1: NEG (12.64/6.12)
|   |   |   |   |   |   |   |   |   CHEMBL1941_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   |   |   CHEMBL2056_agonist_Label-2 <= 1: POS (15.13/3.85)
|   |   |   |   |   |   |   |   |   |   CHEMBL2056_agonist_Label-2 > 1: POS (11.03/5.25)
|   |   |   |   |   |   |   CHEMBL251_non_agonist_Label-2 > 1: POS (13.73/4.26)
|   |   |   |   |   |   CHEMBL1997_non_agonist_Label-2 > 1: NEG (16.64/6.6)
|   |   |   |   CHEMBL3622_non_agonist_Label-2 > 1
|   |   |   |   |   CHEMBL1997_non_agonist_Label-2 <= 1
|   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 <= 1: NEG (36.92/15.92)
|   |   |   |   |   |   CHEMBL211_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 <= 1: POS (10.64/3.57)
|   |   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 > 1: NEG (17.06/7.9)
|   |   |   |   |   CHEMBL1997_non_agonist_Label-2 > 1: NEG (15.98/6.26)
|   |   |   CHEMBL364_non_agonist_Label-2 > 2
|   |   |   |   CHEMBL4303835_non_agonist_Label-2 <= 1
|   |   |   |   |   CHEMBL233_agonist_Label-2 <= 1
|   |   |   |   |   |   CHEMBL1941_agonist_Label-2 <= 1: NEG (31.25/12.5)
|   |   |   |   |   |   CHEMBL1941_agonist_Label-2 > 1
|   |   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 <= 1
|   |   |   |   |   |   |   |   CHEMBL343_non_agonist_Label-2 <= 1: POS (11.21/4.92)
|   |   |   |   |   |   |   |   CHEMBL343_non_agonist_Label-2 > 1: POS (12.03/3.51)
|   |   |   |   |   |   |   CHEMBL216_non_agonist_Label-2 > 1
|   |   |   |   |   |   |   |   CHEMBL206_non_agonist_Label-2 <= 1: NEG (13.03/5.39)
|   |   |   |   |   |   |   |   CHEMBL206_non_agonist_Label-2 > 1: POS (9.06/4.02)
|   |   |   |   |   CHEMBL233_agonist_Label-2 > 1
|   |   |   |   |   |   CHEMBL1941_agonist_Label-2 <= 1: POS (15.59/5.28)
|   |   |   |   |   |   CHEMBL1941_agonist_Label-2 > 1: POS (16.1/7.31)
|   |   |   |   CHEMBL4303835_non_agonist_Label-2 > 1: POS (17.52/4.52)
"""

# Step 1: Parse the text
parsed_nodes = parse_weka_j48_tree(weka_output)

# Step 2: Build the tree structure and assign IDs
root_id, nodes_dict = build_tree_structure(parsed_nodes)

# Step 3: Generate DOT string
dot_output = generate_dot_from_tree(root_id, nodes_dict)

print(dot_output)

# --- How to use the generated DOT output ---
# 1. Save the printed `dot_output` to a file, e.g., `weka_tree.dot`.
# 2. Make sure you have Graphviz installed (as mentioned in the previous answer).
# 3. Open your terminal or command prompt and run:
#    dot -Tpng weka_tree.dot -o weka_tree.png
#    (or -Tsvg, -Tpdf for other formats)
# 4. Open `weka_tree.png` (or your chosen format) to view the tree.