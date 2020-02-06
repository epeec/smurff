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


my ($train, $test, $priors, $config, $line) = ("dead", "dead", "dead", "dead", "dead");

#    std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
#    std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
#    std::shared_ptr<TensorConfig> rowAuxDataDenseMatrixConfig = getRowAuxDataDenseMatrixConfig();
#    std::shared_ptr<TensorConfig> colAuxDataDenseMatrixConfig = getColAuxDataDenseMatrixConfig();
# 
#    Config config = getTestsSmurffConfig();
#    config.setTrain(trainSparseMatrixConfig);
#    config.setTest(testSparseMatrixConfig);
#    config.setPriorTypes({PriorTypes::spikeandslab, PriorTypes::spikeandslab});
#    config.addAuxData({ rowAuxDataDenseMatrixConfig });
#    config.addAuxData({ colAuxDataDenseMatrixConfig });
#    runSession(config, 741);


#   std::shared_ptr<MatrixConfig> trainSparseMatrixConfig = getTrainSparseMatrixConfig();
#   std::shared_ptr<MatrixConfig> testSparseMatrixConfig = getTestSparseMatrixConfig();
#
#   Config config = getTestsSmurffConfig(trainSparseMatrixConfig, testSparseMatrixConfig, {PriorTypes::normal, PriorTypes::normal});


while (<>) {
    $line = $_;

    if (/onfig.setTrain\((\w+)\)/) {
        assert_dead($train, "train");
        $train = $1;
        next; # do not print anything
    }

    if (/onfig.setTest\((\w+)\)/) { 
        assert_dead($test, "test");
        $test = $1;
        next; # do not print anything
    }

    if (/Config (\w+) = getTestsSmurffConfig\(\);/)
    {
        assert_dead($config, "config");
        $config = $1;
        next;
    }

    if (/onfig.setPriorTypes\((.+)\);/) { 
        assert_dead($priors, "priors");
        $priors = $1;
        print("   Config $config = getTestsSmurffConfig($train, $test, $priors);\n");
        ($train, $test, $priors, $config, $line) = ("dead", "dead", "dead", "dead", "dead");
        next; 
    }
 
    print($line)
}
