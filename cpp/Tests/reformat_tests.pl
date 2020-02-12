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

my $found = 0;
my $in_func = 0;
my $values;
my $name;

while (<>) {
    my $line = $_;

    if ($in_func && !$found && /std::vector<double> (\w+) = ({.+});/) {
        $name = $1;
        $values = $2;
        $found = 1;
        next;
    }

    if ($in_func && $found && /$name/)
    {
        $line =~ s/$name/$values/g;
    }

    if ($in_func && /^}/)
    {
        $found = 0;
        $in_func = 0;
    }

    if (/^std::shared_ptr<\w+Config>.+{/)
    {
        $in_func = 1;
    }

    print($line)
}
