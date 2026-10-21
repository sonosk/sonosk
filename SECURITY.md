# Security policy

## Supported versions

Fixes go into the latest release on PyPI. Older versions are not patched.

## Reporting a problem

Please do not open a public issue for a security problem. Use the
private report form instead: on the
[Security tab](https://github.com/sonosk/sonosk/security) of the
repository, choose "Report a vulnerability". Only the maintainer can read
the report. You can also write to soheilkooklan@gmail.com.

Please include the SonoSK version, a short script that shows the problem
and what you expected to happen. You should get an answer within seven
days.

## Errors in safety-related numbers

SonoSK reports acoustic output quantities (mechanical index, derated
intensities) and thermal dose. A calculation error that makes these values
too low is treated with the same priority as a security problem, even
though it is not one in the usual sense, because a user could take a wrong
value at face value. Report it the same way.

SonoSK is research software. It is not a medical device, and its results
must not be used for clinical decisions or for the safety assessment of a
device on the market.
