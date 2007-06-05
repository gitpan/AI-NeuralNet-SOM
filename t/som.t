#########################

# change 'tests => 1' to 'tests => last_test_to_print';

use Test::More qw(no_plan);
BEGIN { use_ok('AI::NeuralNet::SOM') };

######

{
    my $nn = new AI::NeuralNet::SOM (output_dim => "5x6",
				     input_dim  => 3);
    ok ($nn->isa ('AI::NeuralNet::SOM'), 'class');
    is ($nn->{_X}, 5, 'X');
    is ($nn->{_Y}, 6, 'Y');
    is ($nn->{_Z}, 3, 'Z');
    is ($nn->radius, 2.5, 'radius');
}

{
    my $nn = new AI::NeuralNet::SOM (output_dim => "5x6",
				     input_dim  => 3);
    $nn->initialize;
    $nn->train (300, [ 3, 2, 4 ], [ -1, -1, -1 ], [ 0, 4, -3]);

    ok ($nn->as_string, 'pretty print');
    ok ($nn->as_data, 'raw format');

#print $nn->as_string;
}

__END__

