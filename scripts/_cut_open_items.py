import pathlib
p = pathlib.Path('ARXIV_SUBMISSIONS/volume_extended.tex')
t = p.read_text(encoding='utf-8')
start_marker = '\\paragraph{Open items that require new measurement (not new wording).}'
end_marker = 'preregistrations above are commitments to walk it.\n}\n'
# Try alternate end - look for the block end
i = t.find(start_marker)
assert i >= 0, 'start not found'
# Find the closing of "What this preprint is and is not" paragraph
j = t.find('preregistrations above are commitments to walk it.', i)
assert j >= 0, 'end not found'
# advance past the closing brace/newline
k = t.find('\n', j)
# Find next \paragraph or \section to know where to stop cleanly
# Actually just cut from start of "Open items" paragraph through the end of "walk it." line
# Look for end of that paragraph (next blank line)
k = t.find('\n\n', j)
assert k >= 0
new = t[:i] + t[k+2:]
p.write_text(new, encoding='utf-8', newline='\n')
print('removed', k+2-i, 'chars')
print('verify start_marker gone:', start_marker not in new)
print('verify end phrase gone:', 'commitments to walk it' not in new)
