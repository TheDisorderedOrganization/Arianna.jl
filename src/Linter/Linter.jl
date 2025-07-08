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

mutable struct Node
    label::Symbol
    ntype::Symbol
    toptype::Symbol
    children::Vector{Node}
    state::String
end

AbstractTrees.children(n::Node) = n.children


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
    add_policy_node!(policies, pname, ptype=:Policy)

Add a policy node with name `pname` and type `ptype` to the `policies` collection.
The policy node will have its required function children initialized as "Not defined".
"""
function add_policy_node!(policies, pname, ptype=:Policy)
    policy_node = Node(pname, ptype, :Policy, Node[], "✅ Defined")
    # Ajoute les fonctions de policy
    push!(policy_node.children, Node(:sample_action!, :function, :function, Node[], "❌ Not defined"))
    push!(policy_node.children, Node(:log_proposal_density, :function, :function, Node[], " ❌ Not defined"))
    push!(policies, policy_node)
end

"""
    add_action_node!(actions, aname, policies, atype=:Action)

Add an action node with name `aname` and type `atype` to the `actions` collection.
The action node will have its required function children and all given `policies` as children.
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

Add a system node with name `sname` and type `tname` to the `systems` collection.
The system node will have its required function child and all given `actions` as children.
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

Parse the given Julia `code` and return a set of all defined abstract types.
"""
function extract_defined_types(code::String)
    tree = parseall(SyntaxNode, code)
    results = Set{Symbol}()
    collect_defined!(results, tree, :abstract)
    return results
end

"""
    extract_defined_structs(code::String; new_types=nothing) -> Set{Tuple{Symbol, Symbol}}

Parse the given Julia `code` and return a set of tuples (struct_name, struct_type) for all defined structs.
Optionally, `new_types` can be provided to check for additional struct types.
"""
function extract_defined_structs(code::String; new_types=nothing)
    tree = parseall(SyntaxNode, code)
    results = Set{Tuple{Symbol, Symbol}}()
    println(new_types)
    collect_defined!(results, tree, :struct; new_types=new_types)
    return results
end

"""
    extract_defined_functions(code::String) -> Set{Tuple{Symbol, Int, Set{Symbol}}}

Parse the given Julia `code` and return a set of tuples (function_name, arity, argument_types) for all defined functions.
"""
function extract_defined_functions(code::String)
    tree = parseall(SyntaxNode, code)
    results = Set{Tuple{Symbol, Int, Set{Symbol}}}()
    collect_defined!(results, tree, :function)
    return results
end

"""
    collect_defined!(results, node, symb::Symbol; new_types=nothing)

Recursively traverse the syntax tree starting at `node` and collect definitions of the type specified by `symb`
(:abstract, :struct, or :function) into `results`. Optionally, `new_types` can be provided for struct collection.
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

Helper function to collect abstract type definitions from the syntax tree node into `results`.
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

Helper function to collect struct definitions from the syntax tree node into `results`.
Optionally, `new_types` can be provided for additional struct types.
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

Helper function to collect function definitions from the syntax tree node into `results`.
Each function is stored as a tuple (function_name, arity, argument_types).
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

Build the hierarchical tree of systems, actions, policies, and their required functions
from the sets of defined structs, functions, and types.
Returns the root node of the tree.
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
    run_linter(path::String)

Run the linter on the given `path`, which can be a Julia file or a folder.
All Julia files found are concatenated and analyzed together.
Prints the resulting tree structure and linting results.
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