from __future__ import absolute_import, division, print_function, unicode_literals

# standard python
import numpy as np
import scipy
import scipy.spatial
import pathlib
import collections

############# Conversion of a list of atoms to an array of weights and an array derived from views.

# dictionary mapping species name to (row,column) in periodic table
# will need extending for other elements.
# We could include other info about an element here.
# Information about how it bonds within the chemical cannot go here.
speciesdict = {None:(0,0),
               'H':(1,1),
               'B':(2,13), 'C':(2,14), 'N':(2,15), 'O':(2,16),'F':(2,17),
               'Na':(3,1), 'Si':(3,14), 'Al':(3,13), 'P':(3,15), 'S':(3,16), 'Cl':(3,17),
               'K':(4,1), 'Ca':(4,2), 'Ti':(4,4), 'Cr':(4,6), 'Cu':(4,11), 'Zn':(4,12), 'As':(4,15), 'Se':(4,16), 'Br':(4,17),
               'Zr':(5,4), 'Sn':(5,14), 'Sb':(5,15), 'I':(5,17),
               'Pt':(6,10), 'Au':(6,11), 'Hg':(6,12), 'Bi':(6,15)}
def speciesmap(s):
    """Return a np.array version of the species."""
    return np.array(speciesdict[s])


def pendingties(pending,weight=1,oxyz=(False,False,False,False),tol=1e-14,carbonbased=False):
    """Determine if the next stage has ties and if so what they are.
    
    This is an auxiliary routine used by structuretoviews.
    See it for input parameter descriptions.
    :returns: w, tielist.
        w is the weight to be used for each spawned view; it is weight divided by the number of them.
        tielist is a list of indexes within pending that are the start of each spawned view
    
    """
    if not oxyz[0]:
        # just starting, origin not set, full split
        if carbonbased:
            # only 'C'
            tielist = [i for i in range(len(pending)) if pending[i][0]=='C']
            if len(tielist) == 0:
                # no carbon, revert to all
                tielist = range(len(pending))
                print("pendingties: Ignoring carbonbased == True since found only",sorted(collections.Counter([x[0] for x in pending]).items()))
                      #set([x[0] for x in pending]))
        else:
            #everything
            tielist = range(len(pending))
        w = weight / len(tielist)
        return w,tielist
    # Determine if there are ties in the atom closest to the origin
    for i in range(len(pending)):
        # likely a faster way
        if pending[i][2] <= pending[0][2] + tol:
            lasttie = i
    tielist = range(lasttie+1)
    if oxyz[1] and len(tielist) > 1:
        # x-axis set
        # try to break ties using greatest x-coordinate
        maxx = max([pending[i][1][0] for i in tielist])
        tielist = [i for i in tielist if pending[i][1][0] + tol >= maxx]
    if oxyz[2] and len(tielist) > 1:
        # y-axis set
        # try to break ties using greatest y-coordinate
        maxy = max([pending[i][1][1] for i in tielist])
        tielist = [i for i in tielist if pending[i][1][1] + tol >= maxy]
    if oxyz[3] and len(tielist) > 1:
        # z-axis set
        # try to break ties using greatest z-coordinate
        maxz = max([pending[i][1][2] for i in tielist])
        tielist = [i for i in tielist if pending[i][1][2] + tol >= maxz]
        if len(tielist) > 1:
            print("More than one atom at the same position!",[pending[i] for i in tielist])
            raise ValueError
    if len(tielist) == 0:
        print("pendingties: len(tielist)==0 indicates nans")
        print(len(pending),pending,weight,oxyz,tol,carbonbased)
        raise ValueError
    w = weight / len(tielist)
    return w,tielist

def structuretoviews(pending,viewlength=None,weight=1.,done=None,oxyz=(False,False,False,False),tol=1e-14,carbonbased=False):
    """Convert a chemical structure to a set of weighted views.
    
    Operates recursively.
    :parameter pending: List of atoms yet to be included in the view.
        Each atom is of the form a=(t,np.array([x,y,z])) where t is a string atom type like 'C' (or None)
        and (x,y,z) is the coordinates of it. 
        Once the origin is set, a=(t,np.array([x,y,z]),norm((x,y,z))) and the list is sorted by the norm
        from small to large.
    :type pending: list
    :parameter viewlength: The length of views desired. 
        If the structure has too few atoms, then it is filled with null atoms (None,np.array([0.,0.,0.])).
        Left at None, the viewlength will be set to the number of atoms
    :type viewlength: int, >0 or None
    :parameter weight: The weight for the views spawned from here. The total of all weights is 1.
    :type weight: float
    :parameter done: List of atom already in the view.
    :type done: List or None.
    :parameter oxyz: Indicates which parts of the coordinate system have been determined, from
        (origin,xaxis,yaxis,zaxis)
    :type oxyz: tuple of 4 Boolean.
    :parameter tol: A tolerance for deciding when two lengths should be considered the same.
        This should account for the precision of the initial coordinates and later roundoff.
    :type tol: float
    :parameter carbonbased: If True, does the initial split only to have 'C' as the base atoms.
        (If there is no 'C', then reverts to False.)
    :type carbonbased: Boolean
    :returns: A list of (w,v) where w is a scalar and v is a list of atoms of length viewlength.
        The w sum to weight.
    :rtype: list
    
    Note: Consolidation of duplicate views (due to molecule symmetry) is not implemented.
    
    Note: Other info could be included in t as a tuple, should we want to. It is only copied here.
    
    """
    # hardwired tolerance for angle to be considered zero
    angletol = 1e-7 # np.arcos has limited accuracy, giving 1e-8 ish
    if done is None: # start it
        done = []
    if viewlength is None:
        # use all atoms
        viewlength = len(pending)
        if len(done) > 0: # should not happen
            raise ValueError
    # parse for ending conditions
    if viewlength < len(done): # should not happen
        raise ValueError
    if viewlength == len(done): # finished
        #print(done,pending)
        return [(weight,done)]
    if len(pending) == 0: # ran out of atoms,; pad
        return [(weight,done+[(None,np.array([0.,0.,0.])) for i in range(viewlength-len(done))])]
    out = []
    # determine splitting of view needed based on what is known about the coordinates 
    w,tielist = pendingties(pending,weight,oxyz,tol,carbonbased)
    # logic on what to do depends on how much of the coordinate system is determined
    if not oxyz[0]:
        # just starting, origin not set
        if 0 < len(done):  # should not happen
            raise ValueError
        for i in tielist:
            # base atom
            newdone = [(pending[i][0],np.array([0.,0.,0.]))]
            # position that will become origin
            r0 = pending[i][1]
            # other atoms, with new origin
            newpending = [(a[0],a[1]-r0) for a in pending[:i]+pending[i+1:]]
            # add distance from origin information
            newpending = [(a[0],a[1],np.linalg.norm(a[1])) for a in newpending]
            # sort by distance
            newpending = sorted(newpending,key = lambda x: x[2])
            # call recursively to construct
            out.extend(structuretoviews(newpending,viewlength,w,newdone,(True,False,False,False),tol))
    elif not oxyz[1]:
        # origin set but not x-axis
        for i in tielist:
            # loop through ties; could just be one
            # find x-axis and length
            r1 = pending[i][1]
            lenr1 = pending[i][2]
            xaxis = r1/lenr1
            # include the next closest as done, with axis pointing along it
            newdone = done+[(pending[i][0],np.array([lenr1,0.,0.]))]
            # construct the rotation matrix that moves r1 to (lenr1,0,0)
            # compute the angle
            angle = np.arccos(np.inner(xaxis,np.array([1.,0.,0.])))
            if abs(angle) < angletol:
                # Do not rotate
                rotator = scipy.spatial.transform.Rotation.from_rotvec(np.array([0.,0.,0.]))
            else:
                if abs(angle - np.pi ) < angletol: 
                    # close to [-1,0,0]. Use [0,1,0] as rotation axis
                    cross = np.array([0.,1.,0.])
                else:
                    # compute the cross product, normalized
                    cross = np.cross(xaxis,np.array([1.,0.,0.]))
                    cross = cross / np.linalg.norm(cross)
                # make a rotator using rotation vector notation
                rotator = scipy.spatial.transform.Rotation.from_rotvec(angle*cross)
                ## test that rotation is in correct direction
                if np.linalg.norm(rotator.apply(xaxis)-np.array([1.,0.,0.])) > 1e-8:
                    print("rotation error",np.linalg.norm(rotator.apply(xaxis)-np.array([1.,0.,0.])))
                    #raise ValueError
                # other atoms, with new orientation
            newpending = [(a[0],rotator.apply(a[1]),a[2]) for a in pending[:i]+pending[i+1:]]
            # call recursively to construct
            out.extend(structuretoviews(newpending,viewlength,w,newdone,(True,True,False,False),tol))
    elif not oxyz[2]:
        # origin and x-axis set, but not y-axis
        for i in tielist:
            # loop through ties; could just be one
            # test for colinearity, in which case the y-axis cannot be set
            # compute the angle
            cosofangle = pending[i][1][0]/pending[i][2]
            # sometimes can be out of [-1,1] due to roundoff
            if cosofangle >= 1:
                print("structuretoviews: WARNING: cosofangle >= 1:",cosofangle)
                angle = 0.
            elif cosofangle <= -1:
                print("structuretoviews: WARNING: cosofangle <= -1:",cosofangle)
                angle = np.pi
            else:
                angle = np.arccos(cosofangle)
            #print("main angle",angle,abs(angle),abs(angle - np.pi ),angletol)
            if abs(angle) < angletol or abs(angle - np.pi ) < angletol:
                # colinear
                print("colinear",angle)
                # include the next closest as done
                newdone = done+[(pending[i][0],pending[i][1])]
                # other atoms
                newpending = pending[:i]+pending[i+1:]
                # call recursively to construct
                out.extend(structuretoviews(newpending,viewlength,w,newdone,(True,True,False,False),tol))
            else:
                # not colinear, so can determine the y-axis
                lenony = np.linalg.norm(pending[i][1][1:])
                # include the next closest as done, on xy plane
                newdone = done+[(pending[i][0],np.array([pending[i][1][0],lenony,0.]))]
                # find rotation about +/- x-axis that puts current atom onto the xy plane
                # compute the angle
                #cosofangle = pending[i][1][1]/lenony
                angle = np.arccos(pending[i][1][1]/lenony)
                #print('cosofangle, angle =',cosofangle,angle,pending[i][1][1],lenony)
                if pending[i][1][2] >= 0:
                    # Since the z-coordinate is positive, rotate by negative the angle
                    angle = -angle
                rotator = scipy.spatial.transform.Rotation.from_rotvec(angle*np.array([1.,0.,0.]))
                ## test that rotation is in correct direction
                if np.linalg.norm(rotator.apply(pending[i][1])-np.array([pending[i][1][0],lenony,0.])) > 1e-8:
                    print("rotation error",np.linalg.norm(rotator.apply(pending[i][1])-np.array([pending[i][1][0],lenony,0.])))
                    #raise ValueError
                # other atoms, with new orientation
                newpending = [(a[0],rotator.apply(a[1]),a[2]) for a in pending[:i]+pending[i+1:]]
                # call recursively to construct
                # note the z-axis is also set now
                out.extend(structuretoviews(newpending,viewlength,w,newdone,(True,True,True,True),tol))
    elif not oxyz[3]:
        # origin, x-axis, and y-axis set, but not z-axis
        # should not happen
        print("y-axis set but not z-axis, which is nonsense!")
        raise ValueError
    else:
        # coordinate system all set
        if len(tielist) > 1:
            print("More than one atom at the same position!",[pending[i] for i in tielist])
            raise ValueError
        # add on one
        newdone = done+[(pending[0][0],pending[0][1])]
        newpending = pending[1:]
        # call recursively to construct
        out.extend(structuretoviews(newpending,viewlength,w,newdone,(True,True,True,True),tol))
    return out

def vectorizeatomlist(alist,speciesmap):
    """Convert a list of atoms to a vector
    
    :parameter alist: list with a=(t,np.array([x,y,z]))
    :type alist: list
    :parameter speciesmap: function so that speciesmap(t) is an np.array
    :type speciesmap: function
    :returns: vectorization of alist
    :rtype: np.array
    
    """
    vlist = [np.concatenate([speciesmap(a[0]),a[1]]) for a in alist]
    return np.concatenate(vlist)

def matricizeweightsviews(wvlist,speciesmap):
    """Return np.arrays representing the weights and views.
    
    :parameter wvlist: weights and views as constructed by structuretoviews
    :type wvlist: list
    :parameter speciesmap: function so that speciesmap(t) is an np.array,
       where t is a species as written by structuretoviews
    :type speciesmap: function
    :returns: w,v. w is a np.array vector with the weights.
      v is a np.array matrix with the vectorized views. Each row is a view.
    :rtype: np.array,np.array
    
    """
    w = np.array([x[0] for x in wvlist])
    v = np.array([vectorizeatomlist(x[1],speciesmap) for x in wvlist])
    return w,v

###### Handling of pdb a file

def ATOMtoatom(pdbline):
    """
    :parameter pdbline: string from pdb file, must start with 'ATOM'. 
        Assumed formatted like http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
    :type pdbline: string
    :returns: (t,np.array([x,y,z])) where t is the species such as 'Li' 
        and (x,y,z) is the coordinates of it.
    :rtype: tuple
    
    Note: Other info could be included in t as a tuple, should we want to.
    
    """
    # Example: Will give ['ATOM', '1', 'C', 'X', '0', '-0.794', '1.503', '0.558', '0.00', '0.00', 'C']
    pdbline = pdbline.split()
    if pdbline[0] != 'ATOM' and pdbline[0] != 'HETATM' :
        print('Not an atom:',pdbline)
        raise ValueError
    t = pdbline[2]
    # sometime code numbers are included in t; discard them
    t = ''.join([a for a in t if a.isalpha()])
    # For 2-letter atom, make sure first letter capital and second not.
    t = t.capitalize()
    x = float(pdbline[5])
    y = float(pdbline[6])
    z = float(pdbline[7])
    return (t,np.array([x,y,z]))
  
def pdbfiletoatomlist(filename,path=""):
    """Convert a pdb file to a list of atoms.
    
    :parameter filename: the name of the file
    :type filename: string
    :parameter path: The path to the file
    :type path: string
    :returns: list of the atoms, each in the format (t,np.array([x,y,z])),
        as constructed by ATOMtoatom
    :rtype: list
       
    """
    with open(path+filename) as fp:
        lines = fp.readlines()
        out = [ATOMtoatom(x) for x in lines if x[0:4] == 'ATOM' or x[0:6] == 'HETATM']
    return out

def qm7filetowvmats(alist, speciesmap, carbonbased=False, viewlength=None):
    """
    Convert a list of atoms to np.arrays of weights and views for the QM7 dataset.
    
    :parameter alist: a list atoms representing a molecule
    :type alist: list
    :parameter speciesmap: function so that speciesmap(t) is an np.array,
       where t is a species as written by structuretoviews
    :type speciesmap: function
    :parameter carbonbased: If True, only 'C' atoms are considered as the base atoms.
    :type carbonbased: Boolean
    :parameter viewlength: The length of views desired. 
    :type viewlength: int, >0 or None
    :returns: w, v. w is a np.array vector with the weights.
      v is a np.array matrix with the vectorized views. Each row is a view.
    :rtype: np.array, np.array
    """
    wvlist = structuretoviews(alist, viewlength=viewlength, carbonbased=carbonbased)
    #print(wvlist)
    w, v = matricizeweightsviews(wvlist, speciesmap)
    return w, v


#### Bulk handling of files

def find_chemnames(path,extension):
    """Return a list of the chemical names in a directory.

    :parameter path: directory to look in
    :type path: string
    :parameter extension: file name extension to look for, such as '.pdb'
    :type extension: string
    :returns: List of the files in that path that have that extension, not including the extension.
    :rtype: 

    """
    # find all the files and the chemical names or IDs
    p = pathlib.Path(path)
    # full path to the files
    files = [str(x) for x in p.iterdir() if x.is_file()]
    extlen = len(extension)
    plen = len(path)
    chemnames = []
    for f in files:
        if f[-extlen:] == extension:
            # accept and include, stripping path and extension
            chemnames.append(f[plen:-extlen])
        else:
            # reject those without the extension
            print("Rejecting file",f,"without extension",extension)
    return chemnames 


def load_qm7_data(mollist, speciesmap, setNatoms=None, setNviews=None, carbonbased=False, verbose=1):
    species = set()
    Natomslist = [len(mol) for mol in mollist]
    for mol in mollist:
        species.update([atom[0] for atom in mol])
    for s in species:
        try:
            speciesmap(s)
        except:
            print(f"speciesmap could not handle {s}")
            raise NotImplementedError
    
    # Logging information based on the verbose level
    if verbose:
        print(f"{len(mollist)} molecules will be processed.")
        print("Species occurring =", species)
        if verbose > 1:
            print("Number of atoms distribution", sorted(collections.Counter(Natomslist).items()))
        if carbonbased:
            print("Using Carbon based views.")
    # Determine the number of atoms to use for each view
    if setNatoms is None:
        Natoms = max(Natomslist)
        if verbose:
            print("Using max Natoms =", Natoms)
    else:
        Natoms = setNatoms
        if verbose:
            print("Setting all views to Natoms =", Natoms)
    # Process the list of molecules to weights and views
    warrays = []
    varrays = []
    for mol in mollist:
        w, v = qm7filetowvmats(mol, speciesmap, carbonbased, Natoms)
        warrays.append(w)
        varrays.append(v)
    # Find the maximum number of views if not set
    if setNviews is None:
        Nviews = max([v.shape[0] for v in varrays])
    else:
        Nviews = setNviews
        if any(v.shape[0] > Nviews for v in varrays):
            print(f"Insufficient setNviews: {setNviews}, at least one molecule has more views.")
            raise ValueError
    # Create tensors to hold the weights and views data
    Nchem = len(mollist)
    ws = np.zeros((Nchem, Nviews), dtype='float32')
    vectatomspace = max([v.shape[1] for v in varrays])
    vs = np.zeros((Nchem, Nviews, vectatomspace), dtype='float32')
    # Load in the weights and views into the tensors
    for i in range(Nchem):
        ws[i][:len(warrays[i])] = warrays[i]
        vs[i][:varrays[i].shape[0], :varrays[i].shape[1]] = varrays[i]
    if verbose:
        print("Maximum views used =", Nviews)
        print("Data tensor shapes: weights =", ws.shape, ", views =", vs.shape)
    return ws, vs, Natoms, Nviews


##### Computed dependent data

def countspecies(atomlist,speciestocount,proportion=False):
    """Count how many of each species are in this chemical.
    
    :parameter atomlist: List of atoms as returned by pdbfiletoatomlist
    :type atomlist: list
    :parameter speciestocount: List of the species to count. e.g. ['C', 'O', 'S', 'H', 'N'] 
    :type speciestocount: list of strings
    :parameter proportion: If True, divides by the total number of atoms. If the chemical only has 
        species in speciestocount, then the total sums to 1.
    :type proportion: Boolean
    :returns: np.array with the counts or proportions
    :rtype: np.array
    
    """
    out = np.array([len([a for a in atomlist if a[0] == t]) for t in speciestocount])
    if proportion:
        out = out*1./len(atomlist)
    return out

def load_countsfromPDB(path,chemnames,speciestocount,proportion=False,verbose=1):
    """Load (computed) species counts.
    
    :parameter path: root path for this dataset
    :type path: string
    :parameter chemnames: list of the chemical names to get counts for
    :type chemnames: list of strings
    :parameter speciestocount: List of the species to count. e.g. ['C', 'O', 'S', 'H', 'N'] 
    :type speciestocount: list of strings
    :parameter proportion: If True, divides by the total number of atoms. If the chemical only has 
        species in speciestocount, then the total sums to 1.
    :type proportion: Boolean
    :parameter verbose: Print info or not
    :type verbose: Boolean
    :returns: Array with rows corresponding to chemicals and columns to the counts 
    :rtype: np.array
    
    """
    p = path+'PDBs/'
    out = [countspecies(pdbfiletoatomlist(p+c+'.pdb'),speciestocount,proportion=proportion) for c in chemnames]
    if verbose:
        print("For",len(chemnames),"chemicals in",p,"computed counts of",speciestocount)
    # 'float32' for tensorflow
    return np.array(out,dtype='float32')

def chemicaldiameter(atomlist):
    """Compute the diameter of a chemical.
    
    :parameter atomlist: List of atoms as returned by pdbfiletoatomlist
    :type atomlist: list
    :returns: maximum pairwise distance between two atoms
    :rtype: float
    
    """
    out = 0.
    for i in range(len(atomlist)-1):
        for j in range(i+1,len(atomlist)):
            out = max(out,np.linalg.norm(atomlist[i][1]-atomlist[j][1]))
    return out

def load_diametersfromPDB(path,chemnames,verbose=1):
    """Load (computed) chemical diameters.
    
    :parameter path: root path for this dataset
    :type path: string
    :parameter chemnames: list of the chemical names to get counts for
    :type chemnames: list of strings
    :parameter verbose: Print info or not
    :type verbose: Boolean
    :returns: Array with rows corresponding to chemicals and single column with the diameters 
    :rtype: np.array
    
    """
    p = path+'PDBs/'
    out = [[chemicaldiameter(pdbfiletoatomlist(p+c+'.pdb'))] for c in chemnames]
    if verbose:
        print("For",len(chemnames),"chemicals in",p,"computed diameters")
    # 'float32' for tensorflow
    return np.array(out,dtype='float32')

########################################
if __name__ == "__main__":
    if 0:
        ## manual tests of structuretoviews
        s = [("H",np.array([0.1,0.2,0.3])),("C",np.array([1.1,1.2,1.3])),("O",np.array([-1.2,3.4,5.6]))] # simple
        #s = [("H",np.array([0.1,0.2,0.3])),("C",np.array([1.1,0.2,0.3])),("O",np.array([-1.2,3.4,5.6]))] # x-axis already aligned
        #s = [("H",np.array([0.1,0.2,0.3])),("C",np.array([-0.9,0.2,0.3])),("O",np.array([-1.2,3.4,5.6]))] # x-axis opposite
        #s = [("H",np.array([0.1,0.2,0.3])),("C",np.array([1.1,1.2,1.3])),("O",np.array([2.1,2.2,2.3]))] # 3 colinear
        #s = [("H",np.array([0.1,0.2,0.3])),("C",np.array([2.1,1.2,0.3])),("O",np.array([0.1,1.2,2.3]))] # distance tie
        #s = [("H",np.array([0.1,0.2,0.3])),("C",np.array([2.1,1.2,0.3])),("O",np.array([0.1,1.2,2.3])),("N",(5.,7.,-5))] # distance tie, 4
        #s = [("H",np.array([0.1,0.2,0.3])),("N",np.array([1.1,1.2,1.3])),("O",np.array([-1.2,3.4,5.6]))] # simple but no 'C'
        #structuretoviews(s,1)
        #structuretoviews(s,2,tol=1e-14)
        #print(structuretoviews(s,2))
        #structuretoviews(s,4)
        print(structuretoviews(s))
        print(structuretoviews(s,carbonbased=True))
    if 0:
        # manual test of ATOMtoatom
        aline = "ATOM      1  N   ILE     1       0.116  -0.216   0.000  1.00  0.00              "
        aline = "ATOM     14 1HG1 ILE     1       2.259  -2.884  -0.450  1.00  0.00              "
        print(ATOMtoatom(aline))
    if 0:
        #manual test of pdbfiletoatomlist
        path = '../DataFiles/AminoAcid_PDBs/'
        p = pathlib.Path(path)
        files = [x for x in p.iterdir() if x.is_file()]
        species = set()
        for f in files:
            print(f)
            # just testing for exceptions
            x = pdbfiletoatomlist(str(f))
            species.update([a[0] for a in x])
            y = structuretoviews(x,5)
        #  last
        #print('atomlist=',x)
        #print('views=',y)
        #    pdbfiletoatomlist('isoleucine.pdb',path=path)
        print(species)
    if 0:
        # manual test for matricizeweightsviews
        def smap(t): #
            return np.array([999])
        path = '../DataFiles/AminoAcid_PDBs/'
        x = pdbfiletoatomlist('isoleucine.pdb',path=path)
        y = structuretoviews(x,3)
        if 0:
            print(x)
            print(vectorizeatomlist(x,smap))
        if 1:
            print("y=",y)
            print("wout,vout=",matricizeweightsviews(y,smap))
    if 0:
        # manual test for pdbfiletowvmats
        def smap(t): #
            return np.array([999])
        filename = '../DataFiles/AminoAcid_PDBs/isoleucine.pdb'
        w,v = pdbfiletowvmats(filename,smap,carbonbased=True)
        print("w=",w)
        print("v=",v)
    if 0:
        # manual test of load_PDBs
        path = '../DataFiles/Tox21_compounds_firstround/'
        [ws_train, vs_train, Natoms, Nviews, chemnames_train] = load_PDBs(path,carbonbased=True)

    if 0:
        # test of find_chemnames
        path = '../DataFiles/Tox21_all_compounds/PDBs/'
        ext = '.pdb'
        cn = find_chemnames(path,ext)
        print("Found chemnames:",len(cn))
    if 0:
        # test countspecies
        filename = '../DataFiles/Tox21_all_compounds/PDBs/50-22-6.pdb'
        alist = pdbfiletoatomlist(filename)
        speciestocount = ['C', 'O', 'S', 'H', 'N']
        #speciestocount = ['C','O'] # to get learnable labels
        scounts = countspecies(alist,speciestocount,proportion=False)
        scountsp = countspecies(alist,speciestocount,proportion=True)
        print(speciestocount)
        print(scounts,'\n',scountsp)
        print(sum(scountsp))
        print(sorted([a[0] for a in alist]))
    if 0:
        # test load_countsfromPDB
        cn = find_chemnames('../DataFiles/Tox21_all_compounds/PDBs/','.pdb')
        speciestocount = ['C','O']
        path = '../DataFiles/Tox21_all_compounds/'
        countlabels = load_countsfromPDB(path,cn,speciestocount,proportion=False,verbose=1)
        print(countlabels.shape)
        print(countlabels[:2,:])
    if 0:
        # test chemicaldiameter
        from chemdataprep import pdbfiletoatomlist
        filename = '../DataFiles/Tox21_all_compounds/PDBs/50-22-6.pdb'
        alist = pdbfiletoatomlist(filename)
        d = chemicaldiameter(alist)
        print('diameter',d)
    if 1:
        # test load_diametersfromPDB
        cn = find_chemnames('../DataFiles/Tox21_all_compounds/PDBs/','.pdb')
        path = '../DataFiles/Tox21_all_compounds/'
        diamlabels = load_diametersfromPDB(path,cn,verbose=1)
        print(diamlabels.shape)
        print(diamlabels[:2,:])
        
