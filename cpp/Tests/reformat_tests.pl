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

my %mapping = ();
my $in_testcase = 0;

while (<>) {
    my $line = $_;

    if ($in_testcase && /(\w+) = (get\w+AuxData\w+\(\))/) {
        $mapping{$1} = $2;
        next; # do not print anything
    }

    if ($in_testcase && /addAuxData/)
    {
        while (my ($from, $to) = each (%mapping))
        {
            $line =~ s/$from/$to/g;
        }
    }

    if (/TEST_CASE/)
    {
        %mapping = ();
        $in_testcase = 1;
    }

    print($line)
}
