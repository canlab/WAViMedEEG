test -e ssshtest || wget -qhttps://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest

run PEP8 pycodestyle Ex_cnn.py
assert_no_stdout