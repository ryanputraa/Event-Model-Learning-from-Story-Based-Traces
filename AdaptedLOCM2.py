import sys
import itertools
from collections import defaultdict, deque
import logging
import json
import os

# Set up logging
logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(levelname)s - %(message)s')


class FSM:
    def __init__(self):
        self.states = set()
        self.transitions = defaultdict(set)
        self.object_sort = None

    def add_transition(self, from_state, action, to_state):
        self.transitions[from_state].add((action, to_state))
        self.states.update([from_state, to_state])  # Ensure states are added

    def print_fsm(self, file, obj_id):
        if not self.transitions:
            return  # Skip empty FSMs
        file.write(f"FSM for object {obj_id}, sort {self.object_sort}:\n")
        # Sort states based on numerical order
        def state_key(state):
            try:
                return int(state.split('_')[1])
            except (IndexError, ValueError):
                return 0
        for from_state in sorted(self.transitions.keys(), key=state_key):
            for action, to_state in sorted(self.transitions[from_state], key=lambda x: (x[0], state_key(x[1]))):
                file.write(f"  {from_state} --[{action}]--> {to_state}\n")
        file.write("\n")


class LOCM2:
    def __init__(self, output_file):
        self.fsms = defaultdict(list)
        self.output_file = output_file
        self.all_possible_actions = defaultdict(set)  # Now per (obj_id, sort)
        self.transition_matrix = {}  # Transition matrix for FSM construction

    def learn(self, data_list):
        for idx, data in enumerate(data_list):
            logging.info(f"Processing plan {idx + 1}/{len(data_list)}")
            object_sort_sequences = defaultdict(list)  # (obj_id, sort): list of sequences

            sequences = self.parse_plan(data)
            part_of_relations, poss_relations = self.parse_facts(data)

            # For each sequence
            for seq_idx, plan in enumerate(sequences):
                logging.debug(f"Processing sequence {seq_idx + 1}/{len(sequences)}: {plan}")
                obj_sort_to_seq = defaultdict(list)  # (obj_id, sort): sequence of (action, role)
                for action, object_roles in plan:
                    for role, obj_id in object_roles:
                        sorts = self.identify_sorts(obj_id, data, role)
                        logging.debug(f"Identified sorts for object '{obj_id}', role '{role}': {sorts}")
                        for sort in sorts:
                            obj_sort_to_seq[(obj_id, sort)].append((action, role))
                            self.all_possible_actions[(obj_id, sort)].add(action)  # Collect actions per sort
                # Now append the sequences to object_sort_sequences
                for key, seq in obj_sort_to_seq.items():
                    object_sort_sequences[key].append(seq)
                    logging.debug(f"Appended sequence to object '{key[0]}', sort '{key[1]}': {seq}")

            # Process 'part_of' relationships
            for (obj_id, sort), seqs in list(object_sort_sequences.items()):
                ancestors = self.get_all_ancestors(obj_id, part_of_relations)
                logging.debug(f"Object '{obj_id}' has ancestors: {ancestors}")
                for ancestor in ancestors:
                    # Map the sort to the ancestor while preserving role
                    ancestor_sorts = self.map_sort_to_ancestor(sort, ancestor, data)
                    logging.debug(f"Mapped sorts for ancestor '{ancestor}': {ancestor_sorts}")
                    for ancestor_sort in ancestor_sorts:
                        object_sort_sequences[(ancestor, ancestor_sort)].extend(seqs)
                        logging.debug(f"Appended sequences to ancestor '{ancestor}', sort '{ancestor_sort}'")

            # Process 'poss' relationships similarly
            for (obj_id, sort), seqs in list(object_sort_sequences.items()):
                poss_objects = poss_relations.get(obj_id, set())
                logging.debug(f"Object '{obj_id}' has possessed objects: {poss_objects}")
                for poss_obj in poss_objects:
                    # Map the sort to the possessed object while preserving role
                    poss_sorts = self.map_sort_to_ancestor(sort, poss_obj, data)
                    logging.debug(f"Mapped sorts for possessed object '{poss_obj}': {poss_sorts}")
                    for poss_sort in poss_sorts:
                        object_sort_sequences[(poss_obj, poss_sort)].extend(seqs)
                        logging.debug(f"Appended sequences to possessed object '{poss_obj}', sort '{poss_sort}'")


            # Now, for each object and sort, build the FSMs
            for (obj_id, sort), sequences in object_sort_sequences.items():
                if not sequences:
                    continue  # Skip if there are no sequences
                logging.info(f"Building FSMs for object {obj_id}, sort {sort}")

                # Step 1: Split sequences into independent groups
                independent_sequence_groups = self.split_sequences_into_independent_groups(sequences)

                # Remove groups with no actions
                independent_sequence_groups = [
                    group for group in independent_sequence_groups
                    if any(action for seq in group for action, _ in seq)
                ]

                if not independent_sequence_groups:
                    logging.warning(f"No valid sequence groups for object {obj_id}, sort {sort}")
                    continue  # Skip if all groups are empty

                # Step 2: For each independent group, build FSM
                for group_idx, group_sequences in enumerate(independent_sequence_groups, start=1):
                    # Build the transition matrix for this group
                    transition_matrix = self.build_transition_matrix(group_sequences)

                    # Check if the transition matrix is well-formed
                    if self.is_well_formed_matrix(transition_matrix, group_sequences):
                        # Build FSM from the transition matrix
                        fsm = self.build_fsm_from_transitions(group_sequences)
                        if len(independent_sequence_groups) == 1:
                            # Only one group, name it with _all
                            fsm.object_sort = f"{sort}_all"
                        else:
                            # Multiple groups, name them with _partX
                            fsm.object_sort = f"{sort}_part{group_idx}"
                        self.fsms[(obj_id, fsm.object_sort)].append(fsm)
                        logging.debug(f"FSM '{fsm.object_sort}' built for object {obj_id}")
                    else:
                        # If not well-formed, assign as _all even if not well-formed
                        fsm = self.build_fsm_from_transitions(group_sequences)
                        if len(independent_sequence_groups) == 1:
                            fsm.object_sort = f"{sort}_all"
                        else:
                            fsm.object_sort = f"{sort}_part{group_idx}"
                        self.fsms[(obj_id, fsm.object_sort)].append(fsm)
                        logging.debug(f"FSM '{fsm.object_sort}' built (not well-formed) for object {obj_id}")

            # Output FSMs for the current plan
            self.output_fsms_per_plan(idx + 1)

            # Reset FSMs and actions for the next plan
            self.fsms.clear()
            self.all_possible_actions.clear()

    def split_sequences_into_independent_groups(self, sequences):
        """
        Splits sequences into independent groups based on overlapping actions,
        excluding the first action.
        If two sequences share any action beyond the first, they belong to the same group.
        Utilizes the Union-Find algorithm to efficiently group sequences.
        """
        parent = list(range(len(sequences)))  # Initialize parent pointers

        def find(i):
            # Find the root parent of sequence i with path compression
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            # Union the groups of sequences i and j
            pi = find(i)
            pj = find(j)
            if pi != pj:
                parent[pj] = pi

        # Mapping from action to the list of sequence indices that contain this action
        action_to_seq_indices = defaultdict(set)
        for idx, seq in enumerate(sequences):
            # Extract actions, excluding the first one
            actions = set(action for action, _ in seq[1:])
            if not actions:
                # If sequence has only one action, include it
                actions = set(action for action, _ in seq)
            for action in actions:
                action_to_seq_indices[action].add(idx)

        # Perform unions for sequences sharing the same action
        for action, seq_indices in action_to_seq_indices.items():
            seq_indices = list(seq_indices)
            for i in range(1, len(seq_indices)):
                union(seq_indices[0], seq_indices[i])

        # Group sequences by their root parent
        groups = defaultdict(list)
        for idx, seq in enumerate(sequences):
            root = find(idx)
            groups[root].append(seq)

        logging.debug(f"Total independent groups formed: {len(groups)}")
        return list(groups.values())

    def get_all_ancestors(self, obj_id, relations, visited=None):
        if visited is None:
            visited = set()
        ancestors = set()
        for parent in relations.get(obj_id, []):
            if parent not in visited:
                visited.add(parent)
                ancestors.add(parent)
                ancestors.update(self.get_all_ancestors(parent, relations, visited))
        return ancestors

    def map_sort_to_ancestor(self, descendant_sort, ancestor_id, data):
        """
        Maps the sort to the ancestor by generating the ancestor's sorts with the same role.
        """
        # Extract the role from the descendant_sort
        parts = descendant_sort.split('_')
        if len(parts) >= 3:
            role = parts[-1]  # The role is the last part
        else:
            role = 'unknown_role'

        # Identify sorts for the ancestor with this role
        ancestor_sorts = self.identify_sorts(ancestor_id, data, role)
        return ancestor_sorts

    def output_fsms_per_plan(self, plan_number):
        with open(self.output_file, 'a') as file:
            file.write(f"--- Plan {plan_number} ---\n")
            self.print_fsms(file)
            file.write("\n")

    def parse_facts(self, data):
        facts = data.get('facts', [])
        part_of_relations = defaultdict(set)
        poss_relations = defaultdict(set)

        for fact in facts:
            predicate, obj1, obj2 = fact
            if predicate == 'part_of':
                part_of_relations[obj2].add(obj1)  # obj1 is part of obj2
            elif predicate == 'poss':
                poss_relations[obj1].add(obj2)
            else:
                print(f"Unrecognized fact predicate: {predicate}")
        return part_of_relations, poss_relations

    def build_transition_matrix(self, sequences):
        transition_matrix = defaultdict(set)
        for seq in sequences:
            prev_transition = None
            for action_label, _ in seq:
                if prev_transition is not None:
                    transition_matrix[prev_transition].add(action_label)
                prev_transition = action_label
            # Add terminal action with empty set if not already present
            if prev_transition is not None and prev_transition not in transition_matrix:
                transition_matrix[prev_transition] = set()
        return transition_matrix

    def is_well_formed_matrix(self, transition_matrix, group_sequences):
        """
        Determines if the transition matrix is well-formed.
        A well-formed matrix has each action leading to at most one unique follower.
        """
        action_followers = {}
        for seq in group_sequences:
            for i in range(len(seq) - 1):
                action = seq[i][0]
                follower = seq[i + 1][0]
                if action not in action_followers:
                    action_followers[action] = set()
                action_followers[action].add(follower)

        for action, followers in action_followers.items():
            if len(followers) > 1:
                return False  # Action leads to different followers in different contexts
        return True

    def build_fsm_from_transitions(self, group_sequences):
        fsm = FSM()
        state_counter = 0
        initial_state = f"State_{state_counter}"
        fsm.states.add(initial_state)
        state_counter += 1

        # Keep track of states using a mapping from (current_state, action) to next_state
        state_map = {}  # (current_state, action) -> next_state

        # Also keep track of existing states for actions to handle cycles
        action_state_map = {}  # action -> state

        for sequence in group_sequences:
            current_state = initial_state
            for action, role in sequence:
                key = (current_state, action)
                if key in state_map:
                    next_state = state_map[key]
                else:
                    if action in action_state_map:
                        next_state = action_state_map[action]
                    else:
                        next_state = f"State_{state_counter}"
                        state_counter += 1
                        fsm.states.add(next_state)
                        action_state_map[action] = next_state
                    fsm.add_transition(current_state, action, next_state)
                    state_map[key] = next_state
                current_state = next_state
        return fsm

    def identify_sorts(self, obj_id, data, role):
        """
        Identifies and returns a set of sorts for a given object based on its role.
        The sort names are uniquely tied to their respective objects by including the obj_id.
        """
        sorts = set()
        obj_data = data['objects'].get(obj_id)
        if obj_data:
            text, typelist = obj_data
            if typelist:
                highest_prob = -1
                selected_sort = None
                for pos, sd in typelist:
                    if sd:
                        # Find the word sense with the highest probability
                        for prob, word_sense in sd:
                            if prob > highest_prob:
                                highest_prob = prob
                                lemma = word_sense.split('.')[0]
                                selected_sort = f"{obj_id}_{lemma}_{role}"  # Include obj_id
                    else:
                        sort = f"{obj_id}_{pos}_{role}"  # Include obj_id
                        sorts.add(sort)
                if selected_sort:
                    sorts.add(selected_sort)
                elif sorts:
                    pass  # 'sorts' already contains sorts from POS tags
                else:
                    sort = f"{obj_id}_{text}_{role}"  # Include obj_id
                    sorts.add(sort)
            else:
                sort = f"{obj_id}_{text}_{role}"  # Include obj_id
                sorts.add(sort)
        else:
            sorts.add(f'unknown_sort_{role}')
        return sorts

    def parse_plan(self, data):
        events = data['events']
        prec = data.get('prec', [])

        # Build the precedence graph
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        for a, b in prec:
            a = str(a)
            b = str(b)
            graph[a].append(b)
            reverse_graph[b].append(a)

        # Identify initial events (events with no predecessors)
        initial_events = [e for e in events if e not in reverse_graph]
        # Identify final events (events with no successors)
        final_events = [e for e in events if e not in graph]

        # Now generate all paths from initial events to final events
        sequences = []
        for start_event in initial_events:
            paths = self.find_all_paths(graph, start_event, final_events)
            sequences.extend(paths)

        # Now, build the plan sequences
        plan_sequences = []
        for seq_idx, seq in enumerate(sequences):
            logging.debug(f"Processing event sequence {seq_idx + 1}/{len(sequences)}: {seq}")
            plan = []
            for e in seq:
                plan.extend(self.process_event(e, events))
            plan_sequences.append(plan)
        return plan_sequences

    def process_event(self, e_id, events, call_stack_events=None, recursion_warnings=None):
        """
        Processes a single event and returns a list of (action, object_roles) tuples.
        Added cycle detection to prevent infinite recursion.
        """
        plan = []
        if call_stack_events is None:
            call_stack_events = set()
        if recursion_warnings is None:
            recursion_warnings = set()
        if e_id in call_stack_events:
            if e_id not in recursion_warnings:
                logging.warning(f"Detected recursion loop at event {e_id}.")
                recursion_warnings.add(e_id)
            return []
        call_stack_events.add(e_id)

        actual, event = events[e_id]
        action_name = event[0]
        arguments = event[1:]
        object_roles = []  # List of (role, object_id)
        modifiers = {}  # Reset modifiers for each event
        for arg in arguments:
            key, value = arg
            key = key.strip()  # Strip whitespace from key
            if key.startswith('ARGM'):
                # Handle ARGM arguments
                if key == 'ARGM-NEG':
                    modifiers['NEG'] = True
                elif key == 'ARGM-MOD':
                    if isinstance(value, list):
                        # Value may be a list of modal verbs
                        modifiers['MOD'] = '_'.join(value)
                    elif isinstance(value, str):
                        modifiers['MOD'] = value
                elif key in ['ARGM-LOC', 'ARGM-PRP', 'ARGM-DIR', 'ARGM-PNC']:
                    # Locative, purpose, directional adjuncts, or purpose (ARGM-PNC)
                    self.handle_nested_event(value, key, events, plan, object_roles, call_stack_events, recursion_warnings)
                elif key == 'ARGM-COM':
                    # Comitative adjunct, e.g., 'with' someone
                    self.handle_nested_event(value, key, events, plan, object_roles, call_stack_events, recursion_warnings)
                else:
                    # Other ARGM arguments can be handled here
                    logging.warning(f"Encountered unknown ARGM: {key} with value {value}")
            else:
                # Regular argument
                self.handle_nested_event(value, key, events, plan, object_roles, call_stack_events, recursion_warnings)
        # Modify action name based on modifiers
        modified_action_name = action_name
        if 'NEG' in modifiers:
            modified_action_name = 'not_' + modified_action_name
        if 'MOD' in modifiers:
            modified_action_name = modifiers['MOD'] + '_' + modified_action_name
        plan.append((modified_action_name, object_roles))
        call_stack_events.remove(e_id)
        return plan

    def handle_nested_event(self, value, key, events, plan, object_roles, call_stack_events, recursion_warnings):
        """
        Handles nested events in the arguments.
        """
        if isinstance(value, str):
            if value.startswith('E'):
                nested_event_id = value[1:]  # Remove the 'E' prefix
                nested_plan = self.process_event(
                    nested_event_id, events,
                    call_stack_events=call_stack_events,
                    recursion_warnings=recursion_warnings
                )
                plan.extend(nested_plan)
            else:
                object_roles.append((key, value))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            prep, ident = value
            role = f"{key}-{prep}"
            if isinstance(ident, str):
                if ident.startswith('E'):
                    nested_event_id = ident[1:]  # Remove the 'E' prefix
                    nested_plan = self.process_event(
                        nested_event_id, events,
                        call_stack_events=call_stack_events,
                        recursion_warnings=recursion_warnings
                    )
                    plan.extend(nested_plan)
                else:
                    object_roles.append((role, ident))
        else:
            # For other types of values, you may want to handle accordingly
            pass

    def find_all_paths(self, graph, start_event, end_events, path=None, visited=None, max_depth=999):
        if path is None:
            path = []
        if visited is None:
            visited = defaultdict(int)
        path = path + [start_event]
        visited[start_event] += 1
        if start_event in end_events:
            return [path]
        if start_event not in graph:
            return []
        if len(path) > max_depth:
            logging.warning(f"Max depth reached for path: {path}")
            return []
        paths = []
        for node in graph[start_event]:
            if visited[node] < 2:  # Adjust the limit as needed
                newpaths = self.find_all_paths(graph, node, end_events, path, visited.copy(), max_depth)
                for newpath in newpaths:
                    paths.append(newpath)
            else:
                # Optionally, include the path if you want to capture the cycle
                paths.append(path + [node])
        return paths

    def print_fsms(self, file):
        """
        Prints all FSMs to the specified file.
        """
        for (obj_id, sort), fsm_list in sorted(self.fsms.items()):
            for fsm in fsm_list:
                fsm.print_fsm(file, obj_id)


def main():
    # List of JSON files to process
    json_files = [
        'traces_fixed.json',
    ]

    for json_file in json_files:
        if not os.path.isfile('tests/' + json_file):
            print(f"File not found: {json_file}")
            continue  # Skip to the next file

        # Load the data from the JSON file
        with open('tests/' + json_file, 'r') as f:
            data = json.load(f)
            # Ensure data_list is a list
            if isinstance(data, list):
                data_list = data
            else:
                data_list = [data]

        # Determine the output file name by replacing .json with .txt
        base_name = os.path.splitext(json_file)[0]
        output_file = f"output/{base_name}.txt"

        # Clear the output file
        with open(output_file, 'w') as file:
            pass

        locm2 = LOCM2(output_file)
        # Process all plans in the data_list
        locm2.learn(data_list)

        print(f"FSM output written to {output_file}")


# Run the main function
if __name__ == "__main__":
    main()
