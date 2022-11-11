module RRCF
using Base: ident_cmp
import Base.show
using StatsBase
"""
Leaf of RCTree containing no children and at most one parent.
Attributes:
-----------
    i: Index of leaf (user-specified)
    d: Depth of leaf
    u: Pointer to parent
    x: Original point (1 x d)
    n: Number of points in leaf (1 if no duplicates)
    b: Bounding box of point (1 x d)
"""
abstract type Node end


mutable struct Leaf <: Node
const i :: UInt32
    d :: UInt16
    u :: Union{Node, Nothing}
    x :: Vector{Real}
    n :: UInt16
    b 
end
Leaf(; i,d,u,x,n) = Leaf(i,d,u,x,n,[x])
function show(io::IO, obj::Leaf)
    println(io, "Leaf($(obj.i))")
end

"""
Branch of RCTree containing two children and at most one parent.
    q: Dimension of cut
    p: Value of cut
    l: Pointer to left child
    r: Pointer to right child
    u: Pointer to parent
    n: Number of leaves under branch
    b: Bounding box of points under branch (2 x d)
"""
mutable struct Branch <: Node
const   q :: UInt16
const   p :: Float32
        l :: Union{Node, Nothing}
        r :: Union{Node, Nothing}
        u :: Union{Node, Nothing}
        n :: Int32
        b
end
Branch(q,p, u;l=nothing, r=nothing, n=zero(Int32), b=Vector{Vector{Float32}}()) = Branch(UInt16(q),Float32(p),l,r,u,n,b)
function show(io::IO, obj::Branch)
    println(io, "Branch(q=$(obj.q), p=$(obj.p)")
end

"""
Robust random cut tree data structure as described in:
S. Guha, N. Mishra, G. Roy, & O. Schrijvers. Robust random cut forest based anomaly
detection on streams, in Proceedings of the 33rd International conference on machine
learning, New York, NY, 2016 (pp. 2712-2721).
Parameters:
-----------
X: np.ndarray (n x d) (optional)
   Array containing n data points, each with dimension d.
   If no data provided, an empty tree is created.
index_labels: sequence of length n (optional) (default=None)
              Labels for data points provided in X.
              Defaults to [0, 1, ... n-1].
precision: float (optional) (default=9)
           Floating-point precision for distinguishing duplicate points.
random_state: int, RandomState instance or None (optional) (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used by np.random.
Attributes:
-----------
root: Branch or Leaf instance
      Pointer to root of tree.
leaves: dict
        Dict containing pointers to all leaves in tree.
ndim: int
      dimension of points in the tree
Methods:
--------
insert_point: inserts a new point into the tree.
forget_point: removes a point from the tree.
disp: compute displacement associated with the removal of a leaf.
codisp: compute collusive displacement associated with the removal of a leaf
        (anomaly score).
map_leaves: traverses all nodes in the tree and executes a user-specified
            function on the leaves.
map_branches: traverses all nodes in the tree and executes a user-specified
              function on the branches.
query: finds nearest point in tree.
get_bbox: find bounding box of points under a given node.
find_duplicate: finds duplicate points in the tree.
Example:
"""
mutable struct RCTree <: Node
const ndim          :: UInt16 # dimension of points in the tree
      index_labels  :: Vector{UInt32}
      precision     :: Float32
      root          :: Union{Nothing, Branch, Leaf}# Branch or Leaf instance Pointer to root of tree.
      leaves        :: Dict{UInt32, Leaf} # Dict containing pointers to all leaves in tree.
    RCTree(ndim::Int) = begin
        x = new(UInt16(ndim)); 
        x.root = nothing
        x.leaves = Dict{UInt32, Leaf}()
        x
    end
end
function RCTree(X::Vector{Vector{T}}where T; index_labels=nothing, precision=9)
    self = RCTree(length(X))
    self.index_labels = isnothing(index_labels) ? (collect(UInt32, 1:length(X))) : index_labels
    self.leaves = Dict()
    uX = unique(X, dims=1)           # U
    iX = convert(Vector{Int64}, indexin(X, uX))              # I
    nX = counts(iX)                  # N
    if length(uX) == length(X)
        iX = []
    else
        X = uX
    end
    n = length(X)
    mktree(self, X, BitVector(ones(n)), convert(Vector{UInt32},nX), iX, self)
    self.root.u = nothing
    count_all_top_down(self.root)
    get_bbox_top_down(self.root)
    self
end

"""md
Search for leaf nearest to point

    function query(point::Vector{T}where T, node::Union{Branch, Leaf})

Parameters:
-----------
point: np.ndarray (1 x d)
       Point to search for
node: Branch instance
      Defaults to root node
Returns:
--------
nearest: Leaf
         Leaf nearest to queried point in the tree
Example:
--------
# Create RCTree
>>> X = np.random.randn(10, 2)
>>> tree = rrcf.RCTree(X)
# Insert new point
>>> new_point = np.array([4, 4])
>>> tree.insert_point(new_point, index=10)
# Query tree for point with added noise
>>> tree.query(new_point + 1e-5)
Leaf(10)
"""
function query(point::Vector{T}where T, node::Union{Branch, Leaf})::Leaf
    if isa(node, Leaf)
        return node
    else
        point[node.q] <= node.p ? query(point, node.l) : query(point, node.r)
    end
end

"""
Generates the cut dimension and cut value based on the InsertPoint algorithm.

    function insert_point_cut(point::Vector{T}, bbox::Vector{Vector{T}}) where T

Parameters:
-----------
point: np.ndarray (1 x d)
       New point to be inserted.
bbox: np.ndarray(2 x d)
      Bounding box of point set S.
Returns:
--------
cut_dimension: int
               Dimension to cut over.
cut: float
     Value of cut.
Example:
--------
# Generate cut dimension and cut value
>>> _insert_point_cut(x_inital, bbox)
(0, 0.9758881798109296)
"""
function insert_point_cut(point::Vector{T}, bbox::Vector{Vector{T}}) where T
    # Generate the bounding box
    # @assert length(bbox) == 2
    bbox_hat = [min.(bbox[1], point), max.(bbox[end], point)]
    b_span = bbox_hat[end] - bbox_hat[1]
    b_range = reduce(+, b_span)
    r = rand(T)*b_range
    accum = zero(T)
    cut_dimention = UInt16(0)
    while accum < r
        cut_dimention += one(cut_dimention)
        accum += b_span[cut_dimention]
    end
    cut = bbox_hat[1][cut_dimention] + accum - r
    return cut_dimention, cut
end

"""
Called when new point is inserted. Expands bbox of all nodes above new point
if point is outside the existing bbox.
"""
function tighten_bbox_upwards(node::Branch, point::Vector{T} where T)
    while !isnothing(node)
        lt = node.b[1] .> point
        gt = node.b[end] .< point 
        lt_any = any(lt)
        gt_any = any(gt)
        if lt_any || gt_any
            if lt_any
                node.b[1][lt] = point[lt]
            end
            if gt_any
                node.b[end][gt] = point[gt]
            end
        else
            break
        end
        node = node.u
    end
end

"""
Inserts a point into the tree, creating a new leaf
Parameters:
-----------
point: np.ndarray (1 x d)
index: (Hashable type)
       Identifier for new leaf in tree
tolerance: float
           Tolerance for determining duplicate points
Returns:
--------
leaf: Leaf
      New leaf in tree
Example:
--------
# Create RCTree
>>> tree = RCTree()
# Insert a point
>>> x = np.random.randn(2)
>>> tree.insert_point(x, index=0)
"""
function insert_point(tree::RCTree, point::Vector{T}, index::UInt32, tolerance::T=zero(T)):: Leaf where T<:Real
    @assert length(point) == tree.ndim
    if isnothing(tree.root)
        leaf = Leaf(x=point, i=index, u=nothing, d=0, n=1)
        tree.root = leaf
        tree.leaves[index] = leaf
        return leaf
    end
    # Check for existing index in leaves dict
    @assert !haskey(tree.leaves, index)
    # Check for duplicate points
    duplicate = find_duplicate(tree, point, tolerance)
    if !isnothing(duplicate)
        increase_leaf_count_upwards(duplicate)
        tree.leaves[index] = duplicate
        return duplicate
    end
    # If tree has points and point is not a duplicate, continue with main algorithm...
    node = tree.root
    parent = node.u
    maxdepth, _ = findmax(x->x.d, collect(values(tree.leaves)))
    depth = one(UInt16)
    branch = node
    side = :r
    for d in 0:maxdepth
        bbox = node.b
        cut_dimension, cut = insert_point_cut(point, bbox)
        if cut <= bbox[1][cut_dimension]
            leaf = Leaf(x=point, i=index, d=depth, u=nothing,n=1)
            branch = Branch(cut_dimension, cut, nothing, l=leaf, r=node,
                            n=(leaf.n + node.n))
            break
        elseif cut >= bbox[end][cut_dimension]
            leaf = Leaf(x=point, i=index, d=depth, u=nothing, n=1)
            branch = Branch(cut_dimension, cut, nothing, l=node, r=leaf,
                            n=(leaf.n + node.n))
            break
        else
            depth += one(depth)
            if point[node.q] <= node.p
                parent = node
                node = node.l
                side = :l
            else
                parent = node
                node = node.r
                side = :r
            end
        end
    end
    branch.b = lr_branch_bbox(branch)
    # Set parent of new leaf and old branch
    node.u = branch
    leaf.u = branch
    # Set parent of new branch
    branch.u = parent
    if !isnothing(parent)
        # Set child of parent to new branch
        setproperty!(parent, side, branch)
        increase_leaf_count_upwards(parent)
        tighten_bbox_upwards(parent, point)
    else
        # If a new root was created, assign the attribute
        tree.root = branch
    end
    # Increment depths below branch
    map_leaves(node, (x)->x.d+=one(x.d))
    # Add leaf to leaves dict
    tree.leaves[index] = leaf
    # Return inserted leaf for convenience
    return leaf
end
"""
Delete leaf from tree
Parameters:
-----------
index: (Hashable type)
       Index of leaf in tree
Returns:
--------
leaf: Leaf instance
      Deleted leaf
Example:
--------
# Create RCTree
>>> tree = RCTree()
# Insert a point
>>> x = np.random.randn(2)
>>> tree.insert_point(x, index=0)
# Forget point
>>> tree.forget_point(0)
"""
function forget_point(tree::RCTree, index) 
    @assert haskey(tree.leaves, index)
    leaf = tree.leaves[index]
    # If duplicate points exist...
    if leaf.n > 1
        # Simply decrement the number of points in the leaf and for all branches above
        decrease_leaf_count_upwards(leaf)
        return tree.leaves.pop(index)
    end
    # Weird cases here:
    # If leaf is the root...
    if isa(self.root, Leaf) && (leaf === self.root)
        tree.root = nothing
        return tree.leaves.pop(index)
    end
    # Find parent
    parent = leaf.u
    # Find sibling
    if isa(parent.l, Leaf) && (leaf == parent.l)
        sibling = parent.r
    else
        sibling = parent.l
    end
    # If parent is the root...
    if parent === tree.root
        # Set sibling as new root
        sibling.u = nothing
        tree.root = sibling
        # Update depths
        if isa(sibling, Leaf)
            sibling.d = 0
        else
            tree.map_leaves(sibling, op=(x)->x.d-=one(x.d))
        end
        return tree.leaves.pop(index)
    end
    # Find grandparent
    grandparent = parent.u
    # Set parent of sibling to grandparent
    sibling.u = grandparent
    # Short-circuit grandparent to sibling
    if parent === grandparent.l
        grandparent.l = sibling
    else
        grandparent.r = sibling
    end
    # Update depths
    parent = grandparent
    map_leaves(sibling, op=(x)->x.d-=one(x.d))
    # Update leaf counts under each branch
    decrease_leaf_count_upwards(parent)
    # Update bounding boxes
    point = leaf.x
    relax_bbox_upwards(parent, point)
    return self.leaves.pop(index)
end

"""
Compute displacement at leaf
Parameters:
-----------
leaf: index of leaf or Leaf instance
Returns:
--------
displacement: int
              Displacement if leaf is removed
Example:
--------
# Create RCTree
>>> X = np.random.randn(100, 2)
>>> tree = rrcf.RCTree(X)
>>> new_point = np.array([4, 4])
>>> tree.insert_point(new_point, index=100)
# Compute displacement
>>> tree.disp(100)
12
"""
function disp(leaf::Leaf) 
    # Handle case where leaf is root
    if leaf.d === 0
        return 0
    end
    parent = leaf.u
    # Find sibling
    sibling = (leaf === parent.l) ? parent.r : parent.l
    # Count number of nodes in sibling subtree
    sibling.n
end

disp(tree::RCTree, index::Int) = disp(tree.leaves[index])

"""
Compute collusive displacement at leaf
Parameters:
-----------
leaf: index of leaf or Leaf instance
Returns:
--------
codisplacement: float
                Collusive displacement if leaf is removed.
Example:
--------
# Create RCTree
>>> X = np.random.randn(100, 2)
>>> tree = rrcf.RCTree(X)
>>> new_point = np.array([4, 4])
>>> tree.insert_point(new_point, index=100)
# Compute collusive displacement
>>> tree.codisp(100)
31.667
"""
function codisp(leaf::Leaf)
    # Handle case where leaf is root
    if leaf.d === 0
        return 0 
    end
    node = leaf
    results = []
    while !isnothing(node.u)
        parent = node.u
        sibling = (node === parent.l) ? parent.r : parent.l
        result = (sibling.n / node.n)
        push!(results, result)
        node = parent
    end
    co_displacement,_ = findmax(identity, results)
    return co_displacement
end
codisp(tree::RCTree, index::Int) = codisp(tree.leaves[index])

function relax_bbox_upwards(node::Branch, point)
    if !all(broadcast((x,y,z)->x!=y && x!=z, point, node.b[1], node.b[end]))
        node.b = lr_branch_bbox(node)
        relax_bbox_upwards(node.u, point)
    end
end

"""
Traverse tree recursively, calling operation given by op on leaves
Parameters:
-----------
node: node in RCTree
op: function to call on each leaf
*args: positional arguments to op
**kwargs: keyword arguments to op
Returns:
--------
None
Example:
--------
# Use map_leaves to print leaves in postorder
>>> X = np.random.randn(10, 2)
>>> tree = RCTree(X)
>>> tree.map_leaves(tree.root, op=print)
Leaf(5)
Leaf(9)
Leaf(4)
Leaf(0)
Leaf(6)
Leaf(2)
Leaf(3)
Leaf(7)
Leaf(1)
Leaf(8)
"""
function map_leaves(node::Branch, op::Function=(x)->nothing; kwargs...) 
    isnothing(node.l) || map_leaves(node.l, op, kwargs...)
    isnothing(node.r) || map_leaves(node.r, op, kwargs...)
end
map_leaves(node::Leaf, op::Function=()->nothing; kwargs...) = op(node, kwargs...)
"""
Traverse tree recursively, calling operation given by op on branches
Parameters:
-----------
node: node in RCTree
op: function to call on each branch
*args: positional arguments to op
**kwargs: keyword arguments to op
Returns:
--------
None
Example:
--------
# Use map_branches to collect all branches in a list
>>> X = np.random.randn(10, 2)
>>> tree = RCTree(X)
>>> branches = []
>>> tree.map_branches(tree.root, op=(lambda x, stack: stack.append(x)),
                    stack=branches)
>>> branches
[Branch(q=0, p=-0.53),
Branch(q=0, p=-0.35),
Branch(q=1, p=-0.67),
Branch(q=0, p=-0.15),
Branch(q=0, p=0.23),
Branch(q=1, p=0.29),
Branch(q=1, p=1.31),
Branch(q=0, p=0.62),
Branch(q=1, p=0.86)]
"""
function map_branches(node::Branch, op::Function=(x)->nothing; kwargs...) 
    isa(node.l, Branch) && map_branches(node.l, op, kwargs...)
    isa(node.r, Branch) && map_branches(node.r, op, kwargs...)
    op(kwargs...)
end

function get_bbox end
function find_duplicate(tree::RCTree, point::Vector{T}, tolerance=zero(T))where T
    @assert length(point) == tree.ndim
    nearest = query(point, tree.root)
    isapprox(nearest.x, point, atol=tolerance) ? nearest : nothing
end

"""
Create a leaf node from isolated point
"""
function show(io::IO, obj::RCTree)
    depth = ""
    treestr = ""
    function print_push(char_s::Char)
        branch_str = " $(char_s)  "
        depth *= branch_str
    end
    function print_pop()
        depth = depth[collect(eachindex(depth))[1:end-4]]
    end
    function print_tree(node::Union{Leaf, Branch})
        if isa(node, Leaf)
            treestr *= "($(node.i))\n"
        else
            treestr *= "$(Char(9472))+\n"
            treestr *= "$(depth) $(Char(9500))$(Char(9472))$(Char(9472))"
            print_push(Char(9474))
            print_tree(node.l)
            print_pop()
            treestr *= "$(depth) $(Char(9492))$(Char(9472))$(Char(9472))"
            print_push(Char(32))
            print_tree(node.r)
            print_pop()
        end
    end 
    print_tree(::Nothing) = ""
    print_tree(obj.root)
    println(io, treestr)
end

"""
Called after inserting or removing leaves. Updates the stored count of leaves
beneath each branch (branch.n).
"""
function increase_leaf_count_upwards(node::Node)
    node.n += one(node.n)
    isnothing(node.u) || increase_leaf_count_upwards(node.u)
end

function decrease_leaf_count_upwards(node::Node)
    node.n -= one(typeof(node.n))
    isnothing(node.u) || decrease_leaf_count_upwards(node.u)
end

function addleaf(t::RCTree, branch::Branch, X::Vector{Vector{T}}where T, S::BitVector, side::Symbol, depth::UInt16, N::Vector, I::Vector)
    i = findfirst(S)
    leaf = Leaf(i=i, d=depth, u=branch, x=X[i], n=N[i])
    # Link leaf node to parent
    setproperty!(branch, side, leaf)
    println("$(length(S))\t$i")
    # If duplicates exist...
    if length(I) > 0
        # Add a key in the leaves dict pointing to leaf for all duplicate indices
        J = findall(I == i)
        # Get index label
        J = t.index_labels[J]
        for j in J
            t.leaves[j] = leaf
        end
    else
        i = t.index_labels[i]
        t.leaves[i] = leaf
    end
end

function mktree(t::RCTree, X::Vector{Vector{T}}where T, S::BitVector, N::Vector{UInt32}, I, parent::Union{RCTree, Branch}, side=:root, depth::UInt16=UInt16(0))
    # Increment depth as we traverse down
    depth += one(depth)
    # Create a cut according to definition 1
    S1, S2, branch = cut(X, S, parent)

    setproperty!(parent, side, branch)

    # If S1 does not contain an isolated point...
    if count(S1) > 1 
        mktree(t, X, S1, N, I, branch, :l, depth)
    else
        addleaf(t, branch, X, S1, :l, depth, N, I)
    end

    if count(S2) > 1
        mktree(t, X, S2, N, I, branch, :r, depth)
    else
        addleaf(t, branch, X, S2, :r, depth, N, I)
    end
end

function cut(X::Vector{Vector{T}}where T, S::BitVector, parent::Union{Branch, RCTree})
    xmin = [1f35 for x in 1:length(X[1])]
    xmax = [-1f35 for x in 1:length(X[1])]
    for (x,b) in zip(X,S) 
        if b
            xmin = min.(xmin, x)
            xmax = max.(xmax, x)
        end
    end
    l = xmax - xmin
    lsm = reduce(+, l)
    l = l ./ lsm
    # Determine dimension to cut
    q = StatsBase.sample(1:length(X[1]), Weights(l))
    # Determine value for split
    p = rand()* (xmax[q]-xmin[q]) + xmin[q]
    # Determine subset of points to left
    S1 = ([x[q] for x in X] .<= p) .& S #[s && (x[q]<=p) for (x,s) in zip(X,S)]
    # Determine subset of points to right
    S2 = broadcast(~, S1) .& S #map((x,y)->!x && y, S1, S)
    # Create new child node
    child = Branch(q, p, parent)

    return S1, S2, child
end

"""
Recursively compute number of leaves below each branch from
root to leaves.
"""
function count_all_top_down(node::Branch)
    node.n = (isnothing(node.l) ? 0 : (isa(node.l, Leaf) ? node.l.n : count_all_top_down(node.l))) + 
             (isnothing(node.r) ? 0 : (isa(node.r, Leaf) ? node.r.n : count_all_top_down(node.r)))
end

function lr_branch_bbox(node::Branch)
    [min.(node.l.b[1], node.r.b[1]), max.(node.l.b[end], node.r.b[end])]
end

function get_bbox_top_down(node::Branch)
    isa(node.l, Branch) && get_bbox_top_down(node.l)
    isa(node.r, Branch) && get_bbox_top_down(node.r)
    node.b = lr_branch_bbox(node)
end

export 
    # structs
    RCTree,
    # functions
    insert_point,
    forget_point,
    disp,
    codisp,
    map_leaves,
    map_branches,
    query,
    get_bbox,
    find_duplicate
end
