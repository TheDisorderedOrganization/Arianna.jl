module Linter

using JuliaSyntax

export run_lint

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
const POLICY_GUIDED_REQUIRED_FUNCTIONS = [
    (:reward, 2),
]


function extract_defined_functions(code::String)
    tree = parseall(SyntaxNode, code)
    results = Set{Tuple{Symbol, Int}}()
    for node in tree.children
        collect_function_defs!(results, node)
    end
    return results
end

function collect_function_defs!(results::Set, node::SyntaxNode)
    kind = JuliaSyntax.kind(node)

    if Symbol(kind) == :function
        try
            aname = Symbol(node.children[1].children[1].children[1])
            fname = Symbol(node.children[1].children[1].children[2])
            nargs = length(node.children[1].children[2:end])
            if aname == :Arianna
                push!(results, (fname, nargs))
            end
        catch
        end
    end

end

function lint_functions(required_functions::Vector{Tuple{Symbol, Int}}, defined_functions::Set{Tuple{Symbol, Int}}; optional::Bool=false)

    type_functions = optional ? "optional" : "required" 
    missing = filter(r -> !(r in defined_functions), required_functions)

    if !isempty(missing)
        println("❌ Missing $(type_functions) function definitions:")
        for (name, arity) in missing
            println("  - name: $(name) - arity: $(arity)")
        end
    else
        println("✅ All $(type_functions) functions are defined.")
    end
end

function run(file_path::String, policy_guided::Bool=false)
    if !isfile(file_path)
        error("File not found: $file_path")
    end
    defined_functions = extract_defined_functions(read(file_path, String))
    println("Linting file: $file_path")
    
    # Lint required functions
    println("Checking required functions...")
    lint_functions(REQUIRED_FUNCTIONS, defined_functions)

    # Lint additional functions
    println("Checking additional functions...")
    lint_functions(OPTIONAL_FUNCTIONS, defined_functions; optional=true)

    # Lint policy-guided required functions
    if policy_guided
        println("Checking policy-guided required functions...")
        lint_functions(POLICY_GUIDED_REQUIRED_FUNCTIONS, defined_functions)
    end

    println("Linting completed.")
end

end