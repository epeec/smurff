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

    # std::shared_ptr<SideInfoConfig> rowSideInfoDenseMatrix3dConfig = getRowSideInfoDenseMacauPrior3dConfig();
    if ($in_testcase && /(\w+) = (get\w+SideInfo\w+\(\))/) {
        $mapping{$1} = $2;
        next; # do not print anything
    }

    # config.addSideInfoConfig(0, rowSideInfoDenseMatrix3dConfig);
    if ($in_testcase && /(\w+).addSideInfoConfig\((\d+), (\w+)\);/)
    {
        my $config = $1;
        my $mode = $2;
        my $sideinfo = $mapping{$3};
        print("   $config.addSideInfoConfig($mode, $sideinfo);\n");
        next;
    }

    if (/TEST_CASE/)
    {
        %mapping = ();
        $in_testcase = 1;
    }

    print($line)
}
