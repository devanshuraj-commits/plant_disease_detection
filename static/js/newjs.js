$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr('src', e.target.result);
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();
        $('#result').hide();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('.loader').hide();
                $('#btn-predict').show();
                $('#result').fadeIn(600);
                $('#result').text(' Result: ' + data);
                console.log('Success:', data);
            },
            error: function (xhr) {
                $('.loader').hide();
                $('#btn-predict').show();
                $('#result').fadeIn(600);
                var errMessage = xhr.responseText || 'Error predicting image. Please try again.';
                $('#result').text(' Error: ' + errMessage);
                console.error('Error:', errMessage);
            }
        });
    });
});
