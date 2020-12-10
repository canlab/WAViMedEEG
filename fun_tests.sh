test -e ssshtest || wget -qhttps://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest

run PEP8 pycodestyle Ex_cnn.py
assert_no_stdout

run PEP8 pycodestyle Ex_contigs.py
assert_no_stdout

run PEP8 pycodestyle Ex_cleaning.py
assert_no_stdout

run PEP8 pycodestyle Clean.py
assert_no_stdout

run PEP8 pycodestyle Prep.py
assert_no_stdout

run PEP8 pycodestyle ML.py
assert_no_stdout

run PEP8 pycodestyle Standard.py
assert_no_stdout

run PEP8 pycodestyle Signals.py
assert_no_stdout
