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

my @matrixconfig = ();
my $sideconfig = "";
my $priorconfig = "";
my $in_testcase = 0;

while (<>) {
    my $line = $_;
    my $printline = 1;

    # Config matrixSessionConfig = genConfig(trainSparseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal});
    # Config tensorSessionConfig = genConfig(trainSparseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normal});
    # compareSessions(matrixSessionConfig, tensorSessionConfig);

    if ($in_testcase && $line =~ /Config (matrix|tensor)\w+Config = genConfig\((\w+), (\w+), ({.+})\)/) {
        #print(join("\n", $line, $1, $2, $3, $4, $5));
        push @matrixconfig, $2, $3;
        $priorconfig = $4;
        $printline = 0;
    }

    if ($in_testcase && $line =~ /(.addSideInfoConfig.+);/) {
        $sideconfig = $1;
    }

    if ($in_testcase && $line =~ /compareSessions/) {
        # CompareTest(trainDenseMatrix, testSparseMatrix, trainDenseTensor2d, testSparseTensor2d, {PriorTypes::normal, PriorTypes::normal}).runAndCheck();
        $line = "  CompareTest( " . join(", ", @matrixconfig) . ", $priorconfig)$sideconfig.runAndCheck();\n"
    }

    if ($line =~ /TEST_CASE/)
    {
        @matrixconfig = ();
        $priorconfig = "";
        $sideconfig = "";
        $in_testcase = 1;
    }

    if ($printline)
    {
        print($line);
    }
}
