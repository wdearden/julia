# This file is a part of Julia. License is MIT: https://julialang.org/license

module Reqs

import Base: ==
import OldPkg
import ..PkgError
using ..Types

# representing lines of REQUIRE files

abstract type Line end
struct Comment <: Line
    content::AbstractString
end
struct Requirement <: Line
    content::AbstractString
    package::AbstractString
    versions::VersionSet
    system::Vector{AbstractString}

    function Requirement(content::AbstractString)
        fields = split(replace(content, r"#.*$" => ""))
        system = AbstractString[]
        while !isempty(fields) && fields[1][1] == '@'
            push!(system,popfirst!(fields)[2:end])
        end
        isempty(fields) && throw(PkgError("invalid requires entry: $content"))
        package = popfirst!(fields)
        all(field->occursin(Base.VERSION_REGEX, field), fields) ||
            throw(PkgError("invalid requires entry for $package: $content"))
        versions = map(VersionNumber, fields)
        issorted(versions) || throw(PkgError("invalid requires entry for $package: $content"))
        new(content, package, VersionSet(versions), system)
    end
    function Requirement(package::AbstractString, versions::VersionSet, system::Vector{AbstractString}=AbstractString[])
        content = ""
        for os in system
            content *= "@$os "
        end
        content *= package
        if versions != VersionSet()
            for ival in versions.intervals
                (content *= " $(ival.lower)")
                ival.upper < typemax(VersionNumber) &&
                (content *= " $(ival.upper)")
            end
        end
        new(content, package, versions, system)
    end
end

==(a::Line, b::Line) = a.content == b.content
hash(s::Line, h::UInt) = hash(s.content, h + (0x3f5a631add21cb1a % UInt))

# general machinery for parsing REQUIRE files

function read(readable::Vector{<:AbstractString})
    lines = Line[]
    for line in readable
        line = chomp(line)
        push!(lines, occursin(r"^\s*(?:#|$)", line) ? Comment(line) : Requirement(line))
    end
    return lines
end

function read(readable::Union{IO,Base.AbstractCmd})
    lines = Line[]
    for line in eachline(readable)
        push!(lines, occursin(r"^\s*(?:#|$)", line) ? Comment(line) : Requirement(line))
    end
    return lines
end
read(file::AbstractString) = isfile(file) ? open(read,file) : Line[]

function write(io::IO, lines::Vector{Line})
    for line in lines
        println(io, line.content)
    end
end
function write(io::IO, reqs::Requires)
    for pkg in sort!(collect(keys(reqs)), by=lowercase)
        println(io, Requirement(pkg, reqs[pkg]).content)
    end
end
write(file::AbstractString, r::Union{Vector{Line},Requires}) = open(io->write(io,r), file, "w")

function parse(lines::Vector{Line})
    reqs = Requires()
    for line in lines
        if isa(line,Requirement)
            if !isempty(line.system)
                applies = false
                if Sys.iswindows(); applies |=  ("windows"  in line.system); end
                if Sys.isunix();    applies |=  ("unix"     in line.system); end
                if Sys.isapple();   applies |=  ("osx"      in line.system); end
                if Sys.islinux();   applies |=  ("linux"    in line.system); end
                if Sys.isbsd();     applies |=  ("bsd"      in line.system); end
                if Sys.iswindows(); applies &= !("!windows" in line.system); end
                if Sys.isunix();    applies &= !("!unix"    in line.system); end
                if Sys.isapple();   applies &= !("!osx"     in line.system); end
                if Sys.islinux();   applies &= !("!linux"   in line.system); end
                if Sys.isbsd();     applies &= !("!bsd"     in line.system); end
                applies || continue
            end
            reqs[line.package] = haskey(reqs, line.package) ?
                intersect(reqs[line.package], line.versions) : line.versions
        end
    end
    return reqs
end
parse(x) = parse(read(x))

function dependents(packagename::AbstractString)
    pkgs = AbstractString[]
    cd(OldPkg.dir()) do
        for (pkg,latest) in OldPkg.Read.latest()
            if haskey(latest.requires, packagename)
                push!(pkgs, pkg)
            end
        end
    end
    pkgs
end

# add & rm – edit the content a requires file

function add(lines::Vector{Line}, pkg::AbstractString, versions::VersionSet=VersionSet())
    v = VersionSet[]
    filtered = filter(lines) do line
        if !isa(line,Comment) && line.package == pkg && isempty(line.system)
            push!(v, line.versions)
            return false
        end
        return true
    end
    length(v) == 1 && v[1] == intersect(v[1],versions) && return copy(lines)
    versions = reduce(intersect, versions, v)
    push!(filtered, Requirement(pkg, versions))
end

rm(lines::Vector{Line}, pkg::AbstractString) = filter(lines) do line
    isa(line,Comment) || line.package != pkg
end

end # module
