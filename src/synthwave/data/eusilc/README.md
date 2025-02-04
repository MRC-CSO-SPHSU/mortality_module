Current files work only with 2022 data.
All information used here is openly available 
[online](https://circabc.europa.eu/d/a/workspace/SpacesStore/94141a49-a4a7-48bc-89f7-df858c27d016/Methodological%20guidelines%202022%20operation%20v4.pdf).

# Notes on metadata design
## Range
 - single value regardless of its type: use as it is;
 - consecutive category, ordinal or not: employ `Min` and `Max`;
 - category with codes: a list of values;
 - indicators: no need for `DataType` and range, mapping to `True`/`False` might differ
though, see the manual.
 - weights: always positive (?) and `float32`;
 - id: `uint32` unless specified otherwise;
 - income: `float32`, `Max` is always `999999.99`, `Min` is `0` unless specified

## Flag
If there is an accompanying flag for a particular variable the information about
this is contained in the `Extra` section. The default assumption is that
there is at least one flag.
## Dissemination
Some person/household attributes are described in the guidelines,
but no disseminated for privacy reasons. See `Extra` again. 
By default, everything is disseminated.

# TODO add check for degenerate flags
