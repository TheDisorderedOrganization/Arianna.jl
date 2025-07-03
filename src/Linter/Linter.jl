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

const OPTIONAL_FUNCTIONS= [
     (:revert_action!, 2),
]

mutable struct Node
    label::Symbol
    ntype::Symbol
    children::Vector{Node}
    state::String
end

AbstractTrees.children(n::Node) = n.children

function AbstractTrees.printnode(io::IO, n::Node)
    if n.ntype == :function
        print(io, "Function: ", n.label, rpad("", 35 - length(string(n.label))), n.state)
    elseif n.ntype == :AriannaSystem
        printstyled(io, "System: ", n.label, color=:green)
    elseif n.ntype == :Action
        printstyled(io, "Action: ", n.label, color=:red)
    elseif n.ntype == :Policy
        printstyled(io, "Policy: ", n.label, color=:blue)
    end
end

function add_system_tree!(tree, sname, systems)
    sys_node = Node(sname, :AriannaSystem, Node[], "✅ Defined")
    # Ajoute la fonction système
    push!(sys_node.children, Node(:unnormalised_log_target_density, :function, Node[], "❌ Not defined"))
    systems[sname] = sys_node
    push!(tree.children, sys_node)
end

function add_action_system_node!(system_node, aname, actions)
    action_node = Node(aname, :Action, Node[], "✅ Defined")
    # Ajoute les fonctions d'action
    push!(action_node.children, Node(:perform_action!, :function, Node[], "❌ Not defined"))
    push!(action_node.children, Node(:invert_action!, :function, Node[], "❌ Not defined"))
    push!(action_node.children, Node(:revert_action!, :function, Node[], "❕ Not defined"))
    actions[aname] = action_node
    push!(system_node.children, action_node)
end

function add_policy_action_node!(action_node, pname, policies)
    policy_node = Node(pname, :Policy, Node[], "✅ Defined")
    # Ajoute les fonctions de policy
    push!(policy_node.children, Node(:sample_action!, :function, Node[], "❌ Not defined"))
    push!(policy_node.children, Node(:log_proposal_density, :function, Node[], " ❌ Not defined"))
    policies[pname] = policy_node
    push!(action_node.children, policy_node)
end

function extract_defined_structs(code::String)
    tree = parseall(SyntaxNode, code)
    results = Set{Tuple{Symbol, Symbol}}()
    for node in tree.children
        collect_defined_structs!(results, node)
    end
    return results
end

function collect_defined_structs!(results::Set, node::SyntaxNode)
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
            if tname in REQUIRED_STRUCTS
                push!(results, (sname, tname))
            end
        catch
            println(sname_node)
        end
    end
end

function extract_defined_functions(code::String)
    tree = parseall(SyntaxNode, code)
    results = Set{Tuple{Symbol, Int, Set{Symbol}}}()
    for node in tree.children
        collect_function_defs!(results, node)
    end
    return results
end

function collect_function_defs!(results::Set, node::SyntaxNode)
    node_kind = kind(node)
    arg_types = Set{Symbol}()
    if Symbol(node_kind) == :function
        try
            aname = Symbol(node.children[1].children[1].children[1])
            fname = Symbol(node.children[1].children[1].children[2])
            nargs = length(node.children[1].children[2:end])
            for args in node.children[1].children[2:end]
                if !isnothing(args.children)
                    if Symbol(kind(args)) == :(::)
                        push!(arg_types, Symbol(args.children[end]))
                    end
                end
            end
            if aname == :Arianna
                push!(results, (fname, nargs, arg_types))
            end
        catch
        end
    end
end

function construct_tree(defined_structs, defined_functions)
    systems = Dict{Symbol, Node}()
    actions = Dict{Symbol, Node}()
    policies = Dict{Symbol, Node}()
    tree = Node(:root, :root, Node[], "")

    # 1. Ajout des systèmes
    println(defined_structs)
    for (sname, tname) in defined_structs
        if tname == :AriannaSystem
            add_system_tree!(tree, sname, systems)
        end
    end

    # 2. Ajout des actions
    for (aname, tname) in defined_structs
        if tname == :Action
            # Trouver le système parent (ici on suppose que le nom du système est dans le nom de l'action, sinon il faut adapter)
            for sys in values(systems)
                add_action_system_node!(sys, aname, actions)
            end
        end
    end

    # 3. Ajout des policies
    for (pname, tname) in defined_structs
        if tname == :Policy
            for act in values(actions)
                add_policy_action_node!(act, pname, policies)
            end
        end
    end
    # 4. Marquer les fonctions comme définies si présentes
    for (fname, nargs, argtypes) in defined_functions
        for sys in values(systems)
            if sys.label ∉ argtypes
                continue
            end
            for child in sys.children
                if child.label == fname
                    child.state = "✅ Defined"
                    continue
                end
                # Pour les actions
                if child.ntype == :Action
                    if child.label ∉ argtypes
                        continue
                    end
                    for f in child.children
                        if f.label == fname
                            f.state = "✅ Defined"
                            continue
                        end
                    end
                    # Pour les policies
                    for pol in child.children
                        if pol.ntype == :Policy
                            if pol.label ∉ argtypes
                                continue
                            end
                            for fpol in pol.children
                                if fpol.label == fname
                                    fpol.state = "✅ Defined"
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    return tree
end


function run(path::String, policy_guided::Bool=false)
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
    defined_structs = extract_defined_structs(full_code)
    defined_functions = extract_defined_functions(full_code)
    tree = construct_tree(defined_structs, defined_functions)
    println("Linting : $path")
    print_tree(tree)
end

end