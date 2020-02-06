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

while (<>) {
    my $line = $_;

    if (/(\w+) = (getTrain\w+\(\))/) {
        $mapping{$1} = $2;
        next; # do not print anything
    }

    if (/(\w+) = (getTest\w+\(\))/) {
        $mapping{$1} = $2;
        next; # do not print anything
    }

    # Config tensorRunConfig = getTestsSmurffConfig(trainDenseTensorConfig, testSparseTensorConfig, {PriorTypes::macau, PriorTypes::macau});
    if (/Config (\w+) = getTestsSmurffConfig\((\w+), (\w+), (.+)\);/)
    {
        my $config = $1;
        my $train = $mapping{$2};
        my $test = $mapping{$3};
        my $priors = $4;
        print("   Config $config = getTestsSmurffConfig($train, $test, $priors);\n");
        next;
    }

    if (/TEST_CASE/)
    {
        %mapping = ();
    }

    print($line)
}
