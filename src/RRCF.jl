module RRCF
using Base: ident_cmp
using StatsBase
using JSON3
import Base.show
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

struct Leaf <: Node
    i :: UInt32
    d :: UInt16
    u 
    x 
    n
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
        n :: UInt32
        b
end
Branch(q,p, u;l=nothing, r=nothing, n=zero(UInt32), b=zero(UInt16)) = Branch(UInt16(q),Float32(p),l,r,u,n,b)
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
      X             # d x N array
      index_labels  :: Vector{UInt32}
      precision     :: Float32
      root          :: Node# Branch or Leaf instance Pointer to root of tree.
      leaves        :: Dict # Dict containing pointers to all leaves in tree.
    RCTree(ndim::Int) = begin
        x = new(UInt16(ndim)); 
        x.root = x
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
    mktree(self, X, ones(Bool, n), nX, iX, self.root)
    self.root.u = nothing
    count_all_top_down(self.root)
    get_bbox_top_down(self.root)
    self
end

function insert_point end
function forget_point end
function disp end
function codisp end
function map_leaves end
function map_branches end
function query end
function get_bbox end
function find_duplicate end

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
    print_tree(obj.root)
    println(io, treestr)
end

function addleaf(t::RCTree, branch::Branch, X::Vector{Vector{T}}where T, S::Vector{Bool}, side::Symbol, depth::UInt16, N::Vector, I::Vector)
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

function mktree(t::RCTree, X, S, N, I, parent::Union{RCTree, Branch}, side=:root, depth::UInt16=UInt16(0))
    # Increment depth as we traverse down
    depth += one(depth)
    # Create a cut according to definition 1
    S1, S2, branch = cut(X, S, parent, side)
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

function cut(X::Vector{Vector{T}}where T, S::Vector{Bool}, parent::Union{Branch, RCTree}, side::Symbol=:l)
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
    S1 = [s && (x[q]<=p) for (x,s) in zip(X,S)]
    # Determine subset of points to right
    S2 = map((x,y)->!x && y, S1, S)
    # Create new child node
    child = Branch(q, p, parent)
    # Link child node to parent
    if !isa(parent, Nothing)
        setproperty!(parent, side, child)
    end
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


end
