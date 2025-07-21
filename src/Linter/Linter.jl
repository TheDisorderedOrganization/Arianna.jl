"""
    Linter

Module for linting user's code in the Arianna framework.
It checks for the presence of required structs, functions, and their definitions.
"""
module Linter

using JuliaSyntax
using AbstractTrees
export run

const REQUIRED_STRUCTS = [
    :AriannaSystem,
    :Action,
    :Policy,
]

const REQUIRED_FUNCTIONS = [
    (:sample_action!, 5),
    (:log_proposal_density, 4),
    (:perform_action!, 2),
    (:unnormalised_log_target_density, 2),
    (:invert_action!, 2),   
]

const OPTIONAL_FUNCTIONS = [
     (:revert_action!, 2),
]

"""
    Node

Represents a node in the hierarchical Arianna system tree. Each node can represent a system, action,
policy, or function, and may contain child nodes.

# Fields
- `label::Symbol`: The name or identifier of the node (e.g., function or struct name).
- `ntype::Symbol`: The type of the node (e.g., `:System`, `:Action`, `:Policy`, or `:Function`).
- `toptype::Symbol`: The top-level category the node belongs to (typically used for system type inheritance).
- `children::Vector{Node}`: A list of child `Node`s representing the structure beneath this node.
- `state::String`: The linting state of the node (e.g., `"✅ Defined"` or `"❌ Missing"`).

# Usage
Used to build and represent the structured tree for static analysis and linting of Arianna-based systems.
"""
mutable struct Node
    label::Symbol
    ntype::Symbol
    toptype::Symbol
    children::Vector{Node}
    state::String
end

AbstractTrees.children(n::Node) = n.children


"""
    AbstractTrees.printnode(io::IO, n::Node)

Prints a formatted representation of a `Node` object to the given IO stream.

Depending on the `toptype` of the node, the output is styled and labeled accordingly:
- If `toptype` is `:function`, prints the function label and its state, padded for alignment.
- If `toptype` is `:AriannaSystem`, prints the system label in green.
- If `toptype` is `:Action`, prints the action label in red.
- If `toptype` is `:Policy`, prints the policy label in blue.

# Arguments
- `io::IO`: The IO stream to print to.
- `n::Node`: The node to be printed.
"""
function AbstractTrees.printnode(io::IO, n::Node)
    if n.toptype == :function
        print(io, "Function: ", n.label, rpad("", 35 - length(string(n.label))), n.state)
    elseif n.toptype  == :AriannaSystem
        printstyled(io, "System: ", n.label, color=:green)
    elseif n.toptype == :Action
        printstyled(io, "Action: ", n.label, color=:red)
    elseif n.toptype == :Policy
        printstyled(io, "Policy: ", n.label, color=:blue)
    end
end

"""
    add_policy_node!(policies, pname; ptype=:Policy)

Add a new policy node to the `policies` collection.

# Arguments
- `policies`: The collection (e.g., dictionary or custom structure) to which the policy node will be added.
- `pname`: The name (identifier) of the policy node.
- `ptype`: The type of the policy node (default is `:Policy`).

# Behavior
Initializes the policy node with its required function children set to `"Not defined"`.

# Returns
Modifies `policies` in-place by adding the new policy node.
"""
function add_policy_node!(policies, pname, ptype=:Policy)
    policy_node = Node(pname, ptype, :Policy, Node[], "✅ Defined")
    # Ajoute les fonctions de policy
    push!(policy_node.children, Node(:sample_action!, :function, :function, Node[], "❌ Not defined"))
    push!(policy_node.children, Node(:log_proposal_density, :function, :function, Node[], " ❌ Not defined"))
    push!(policies, policy_node)
end

"""
    add_action_node!(actions, aname, policies; atype=:Action)

Adds a new action node to the `actions` collection.

# Arguments
- `actions`: The collection to which the new action node will be added.
- `aname`: The name of the action node.
- `policies`: A list of policy nodes to attach as children to the action node.
- `atype`: (Optional) The type of the node, defaults to `:Action`.

# Behavior
Creates an action node with the specified name and type, attaches all required function children, and adds the provided `policies` as children nodes.

# Returns
Modifies `actions` in-place by adding the new action node.
"""
function add_action_node!(actions, aname, policies, atype=:Action)
    action_node = Node(aname, atype, :Action, Node[], "✅ Defined")
    # Ajoute les fonctions d'action
    push!(action_node.children, Node(:perform_action!, :function, :function, Node[], "❌ Not defined"))
    push!(action_node.children, Node(:invert_action!, :function, :function, Node[], "❌ Not defined"))
    push!(action_node.children, Node(:revert_action!, :function, :function, Node[], "❕ Not defined"))
    push!(action_node.children, Node(:reward, :function, :function, Node[], "❕ Not defined"))
    for policy in policies
        push!(action_node.children, policy)
    end
    push!(actions, action_node)
end

"""
    add_system_node!(systems, sname, actions; tname=:AriannaSystem)

Add a new system node to the `systems` collection.

# Arguments
- `systems`: The collection to which the system node will be added.
- `sname`: Symbol or string specifying the name of the system node.
- `actions`: A collection of actions to be added as children of the system node.
- `tname`: (Optional) Symbol specifying the type of the system node. Defaults to `:AriannaSystem`.

# Behavior
Creates a system node with the specified name and type, attaches a required function child, and adds all provided actions as children nodes.

# Returns
Modifies `systems` in-place by adding the new system node.
"""
function add_system_node!(systems, sname, actions; tname=:AriannaSystem)
    system_node = Node(sname, tname, :AriannaSystem, Node[], "✅ Defined")
    # Ajoute la fonction système
    push!(system_node.children, Node(:unnormalised_log_target_density, :function, :function, Node[], "❌ Not defined"))
    for action in actions
        push!(system_node.children, action)
    end
    push!(systems, system_node)
end

"""
    extract_defined_types(code::String) -> Set{Symbol}

Analyze the provided Julia source code and return a set containing the names of all abstract types defined within it.

# Arguments
- `code::String`: A string containing Julia source code to be parsed.

# Returns
- `Set{Symbol}`: A set of symbols, each representing the name of an abstract type defined in the code.
"""
function extract_defined_types(code::String)
    tree = parseall(SyntaxNode, code)
    results = Set{Symbol}()
    collect_defined!(results, tree, :abstract)
    return results
end

"""
    extract_defined_structs(code::String; new_types=nothing) -> Set{Tuple{Symbol, Symbol}}

Parses the provided Julia source `code` and returns a set of tuples `(struct_name, struct_type)` for each struct definition found.

- `struct_name`: The name of the struct as a `Symbol`.
- `struct_type`: The type of the struct as a `Symbol` (e.g., `:mutable`, `:struct`, or any custom type in `new_types`).

# Arguments
- `code::String`: The Julia source code to analyze.
- `new_types`: (optional) An iterable of additional struct types to recognize beyond the default `struct` and `mutable struct`.

# Returns
- `Set{Tuple{Symbol, Symbol}}`: A set containing tuples of struct names and their corresponding types.
"""
function extract_defined_structs(code::String; new_types=nothing)
    tree = parseall(SyntaxNode, code)
    results = Set{Tuple{Symbol, Symbol}}()
    collect_defined!(results, tree, :struct; new_types=new_types)
    return results
end

"""
    extract_defined_functions(code::String) -> Set{Tuple{Symbol, Int, Set{Symbol}}}

Parses the provided Julia source `code` and returns a set of tuples describing each function definition found.

- `function_name`: The name of the function as a `Symbol`.
- `arity`: The number of arguments the function takes (as an `Int`).
- `arg_types`: A `Set{Symbol}` representing the types of the arguments, if specified; otherwise, the set may be empty.

# Arguments
- `code::String`: The Julia source code to analyze.

# Returns
- `Set{Tuple{Symbol, Int, Set{Symbol}}}`: A set containing tuples of function names, arities, and sets of argument type symbols.
"""
function extract_defined_functions(code::String)
    tree = parseall(SyntaxNode, code)
    results = Set{Tuple{Symbol, Int, Set{Symbol}}}()
    collect_defined!(results, tree, :function)
    return results
end

"""
    collect_defined!(results, node, symb::Symbol; new_types=nothing)

Recursively traverses the syntax tree starting at `node` and collects definitions of the specified kind into `results`.

- `symb`: The type of definitions to collect (`:abstract`, `:struct`, or `:function`).
- `new_types`: Optional set of additional struct types to recognize when `symb == :struct`.

# Arguments
- `results`: A mutable collection to store the collected definitions.
- `node`: A `SyntaxNode` from CSTParser representing the current location in the syntax tree.
- `symb::Symbol`: The kind of definition to collect.
- `new_types`: (optional) Additional struct types beyond the default ones.

# Returns
- Nothing. The function modifies `results` in place.
"""
function collect_defined!(results, node, symb::Symbol; new_types=nothing)
    
    if symb == :abstract    
        collect_defined_types!(results, node)
    
    elseif symb == :function
        collect_defined_functions!(results, node)
    
    elseif symb == :struct
        collect_defined_structs!(results, node, new_types=new_types)
    end

    if isnothing(node.children)
        return
    end

    for child in node.children
        collect_defined!(results, child, symb, new_types=new_types)
    end
end

"""
    collect_defined_types!(results::Set, node)

Helper function that collects abstract type definitions from the given `node` into `results`.

Only types that inherit from `AriannaSystem` are collected.

# Arguments
- `results::Set`: A mutable set to store the collected abstract type names.
- `node`: A `SyntaxNode` to examine.

# Returns
- Nothing. The function modifies `results` in place.
"""
function collect_defined_types!(results::Set, node)
    node_kind = kind(node)
    if Symbol(node_kind) == :abstract
        try
            tname = Symbol(node.children[1].children[1]) 
            ttname = Symbol(node.children[1].children[2])
            if Symbol(ttname) == :AriannaSystem
                push!(results, tname)
            end
            return
        catch
            return
        end

    end
    return
end

"""
    collect_defined_structs!(results, node::SyntaxNode; new_types=nothing)

Helper function that collects struct definitions from the given `node` into `results`.

Structs are included only if their declared type is found in `REQUIRED_STRUCTS` or in the optional `new_types` set.

# Arguments
- `results`: A mutable set to store `(struct_name, struct_type)` tuples.
- `node::SyntaxNode`: The node to examine.
- `new_types`: (optional) A set of additional struct types to recognize.

# Returns
- Nothing. The function modifies `results` in place.
"""
function collect_defined_structs!(results, node::SyntaxNode; new_types=nothing)
    node_kind = kind(node)
    if Symbol(node_kind) == :struct
        # Extraction robuste du nom du struct
        sname_node = node.children[1]
        try
            if !isnothing(sname_node.children) 
                sname_sub = sname_node.children[1]
                if !isnothing(sname_sub.children) 
                    sname = Symbol(sname_sub.children[1])
                else
                    sname = Symbol(sname_sub)
                end
            end
            tname = Symbol(sname_node.children[end])
            if tname ∈ REQUIRED_STRUCTS || tname ∈ new_types
                push!(results, (sname, tname))
            end
            return
        catch
            return
        end
    end
    return
end

"""
    collect_defined_functions!(results, node::SyntaxNode)

Helper function that collects function definitions from the given `node` into `results`.

Functions are only collected if they belong to `Arianna` or `Arianna.PolicyGuided` namespaces.

Each entry is stored as a tuple:
- `function_name`: The name of the function as a `Symbol`.
- `arity`: The number of arguments.
- `arg_types`: A `Set{Symbol}` of the argument type names, if specified.

# Arguments
- `results`: A mutable set to store `(function_name, arity, arg_types)` tuples.
- `node::SyntaxNode`: The node to examine.

# Returns
- Nothing. The function modifies `results` in place.
"""
function collect_defined_functions!(results, node::SyntaxNode)
    node_kind = kind(node)
    arg_types = Set{Symbol}()
    if Symbol(node_kind) != :function
        return
    end
    fn_header = node.children[1]
    fn_name_expr = fn_header.children[1]

    aname1 = nothing
    aname2 = nothing
    fname = nothing

    # Extract function name parts
    children = fn_name_expr.children
    try
        if length(children) == 2
            if !isnothing(children[1].children)
                aname1 = Symbol(children[1].children[1])
                aname2 = Symbol(children[1].children[2])
                fname = Symbol(children[2])
            else
                aname1 = Symbol(children[1])
                fname = Symbol(children[2])
            end
        end
        nargs = length(fn_header.children[2:end])
        for args in fn_header.children[2:end]
            if !isnothing(args.children)
                if Symbol(kind(args)) == :(::)
                    push!(arg_types, Symbol(args.children[end]))
                end
            end
        end
        if aname1 == :Arianna
            if isnothing(aname2) || aname2 == :PolicyGuided
                push!(results, (fname, nargs, arg_types))
            end
        end
    catch
    end
end

"""
    construct_tree(defined_structs, defined_functions, defined_types) -> Node

Builds a hierarchical tree representing the structure of an Arianna system from the provided sets
of defined structs, functions, and abstract types.

Each system node may contain actions, which in turn may contain policies, each requiring specific functions.
The function cross-references the required functions with the provided `defined_functions` and marks
them as defined if present.

# Arguments
- `defined_structs`: A set of tuples `(struct_name::Symbol, struct_type::Symbol)` representing defined structs.
- `defined_functions`: A set of tuples `(function_name::Symbol, arity::Int, arg_types::Set{Symbol})` representing defined functions.
- `defined_types`: A set of `Symbol`s representing additional abstract system types.

# Returns
- `Node`: The root node of the constructed tree.
"""
function construct_tree(defined_structs, defined_functions, defined_types)
    systems = Set{Node}()
    actions = Set{Node}()
    policies = Set{Node}()
    tree = Node(:root, :root, :root, Node[], "")

    for (pname, ptname) in defined_structs
        if ptname == :Policy
            add_policy_node!(policies, pname)
        end
    end

    for (aname, atname) in defined_structs
        if atname == :Action
            add_action_node!(actions, aname, deepcopy(policies))
        end
    end

    for (sname, stname) in defined_structs
        if stname == :AriannaSystem
            add_system_node!(systems, sname, deepcopy(actions))
        end
    end

    for tname in defined_types
        for (sname, stname) in defined_structs
            if stname == tname
                add_system_node!(systems, sname, deepcopy(actions), tname=tname)
            end
        end
    end
        
    for system in systems
        push!(tree.children, system)
    end

   for (fname, _, argtypes) in defined_functions
        for system in tree.children
            (system.label ∈ argtypes || system.ntype ∈ argtypes || :AriannaSystem ∈ argtypes) || continue

            for faction in system.children
                if faction.label == fname
                    faction.state = "✅ Defined"
                    continue
                end

                faction.ntype == :Action || continue
                (faction.label ∈ argtypes || :Action ∈ argtypes) || continue

                for func in faction.children
                    func.label == fname && (func.state = "✅ Defined")
                end

                for policy in faction.children
                    policy.ntype == :Policy || continue
                    (policy.label ∈ argtypes || :Policy ∈ argtypes) || continue

                    for fpolicy in policy.children
                        fpolicy.label == fname && (fpolicy.state = "✅ Defined")
                    end
                end
            end
        end
    end

    return tree
end

"""
    construct_tree(defined_structs, defined_functions, defined_types) -> Node

Builds a hierarchical tree representing the structure of an Arianna system from the provided sets
of defined structs, functions, and abstract types.

Each system node may contain actions, which in turn may contain policies, each requiring specific functions.
The function cross-references the required functions with the provided `defined_functions` and marks
them as defined if present.

# Arguments
- `defined_structs`: A set of tuples `(struct_name::Symbol, struct_type::Symbol)` representing defined structs.
- `defined_functions`: A set of tuples `(function_name::Symbol, arity::Int, arg_types::Set{Symbol})` representing defined functions.
- `defined_types`: A set of `Symbol`s representing additional abstract system types.

# Returns
- `Node`: The root node of the constructed tree.
"""
function run_linter(path::String)
    files = String[]
    if isfile(path) && endswith(path, ".jl")
        push!(files, path)
    elseif isdir(path)
        files = filter(f -> endswith(f, ".jl"),
                       reduce(vcat, [joinpath(root, f) for (root, _, files) in walkdir(path) for f in files]))
    else
        error("The file or the folder doesn't exist")
    end
    # Concatène tout le code
    full_code = join(read.(files, String), "\n")
    # Analyse globale
    defined_types = extract_defined_types(full_code)
    defined_structs = extract_defined_structs(full_code; new_types=defined_types)
    defined_functions = extract_defined_functions(full_code)
    tree = construct_tree(defined_structs, defined_functions, defined_types)
    println("Linting : $path")
    print_tree(tree)
end

end