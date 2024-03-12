_PQR_ATOM_FORMAT_STRING = (
    "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f %7s  %6s      %2s\n"
)

_TER_FORMAT_STRING = (
    "TER   %5i      %3s %c%4i%c                                                      \n"
)

def raw_info_to_pqr_block(coords, elements, charges=None, radii=None):
    pqr_block = ''
    
    if charges is None:
        charges = [0.0] * len(coords)
    if radii is None:
        radii = [0.0] * len(coords)
    
    for i, (coord, element, charge, radius) in enumerate(zip(coords, elements, charges, radii)):
        x, y, z = coord[0], coord[1], coord[2]
        element = element

        record_type = 'HETATM'
        atom_number = i
        name = element
        altloc = ' '
        resname = 'UNK'
        chain_id = 'L'
        resseq = i
        icode = ' '
        x = x
        y = y
        z = z
        pqr_charge = charge
        radius = radius
        element = element

        args = (
            record_type,
            atom_number,
            name,
            altloc,
            resname,
            chain_id,
            resseq,
            icode,
            x,
            y,
            z,
            pqr_charge,
            radius,
            element,
        )
        curr_line = _PQR_ATOM_FORMAT_STRING % args
        pqr_block += curr_line
    ter_line = _TER_FORMAT_STRING %(i, 'UNK', 'L', i, ' ')
    pqr_block += ter_line
    end_line = 'END\n'
    pqr_block += end_line
    return pqr_block
