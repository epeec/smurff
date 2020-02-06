#!/usr/bin/env perl
#
#
use warnings;
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

my @lines = ();
while (<>) {
    push @lines, $_;
}

my %mapping = ();
for (my $lineno=0; $lineno <= $#lines ; $lineno++) 
{

    my $line = $lines[$lineno];

    # Config tensorRunConfig = getTestsSmurffConfig(trainDenseTensorConfig, testSparseTensorConfig, {PriorTypes::macau, PriorTypes::macau});
    if ($line =~ /Config (\w+) = getTestsSmurffConfig\((.+)\);/)
    {
        my $config = $1;
        my $params = $2;
        print("   Config $config = getTestsSmurffConfig($params)");

        while ($lines[$lineno+1] =~ /addSideInfoConfig\((.+)\);/)
        {
            print(".addSideInfoConfig($1)");
            $lineno++;
        }
        
        print(";\n");
    }
    else
    {
        print($line);
    }

    if ($line =~ /TEST_CASE/)
    {
        %mapping = ();
    }

}
