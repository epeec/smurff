#!/usr/bin/env perl
#
#

use strict;

sub assert_dead
{
    my ($value, $name) = @_;
    if ($value ne "dead")
    {
        print "//$name not dead ($value)\n";
    }
    else
    {
        #print "//$name dead ($value)\n";
    }
}

my $config = "";
my $in_testcase = 0;

while (<>) {
    my $line = $_;

    if ($in_testcase && /Config config = (genConfig.+);/) {
        $config = $1;
        next;
    }

    if ($in_testcase && /runAndCheck/)
    {
        $line =~ s/config/$config/g;
    }

    if (/TEST_CASE/)
    {
        $config = "";
        $in_testcase = 1;
    }

    print($line)
}
