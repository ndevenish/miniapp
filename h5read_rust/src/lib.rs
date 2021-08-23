use libc::c_char;
use std::ffi::CStr;

pub struct H5File {
    filename: String,
    master_file: hdf5::File,
}

#[no_mangle]
pub extern "C" fn h5read_open(filename: *const c_char) -> *mut H5File {
    let c_filename = unsafe {
        assert!(!filename.is_null());
        CStr::from_ptr(filename)
    };
    let r_filename = c_filename.to_str().unwrap().to_owned();
    if let Ok(file) = hdf5::File::open(&r_filename) {
        Box::into_raw(Box::new(H5File {
            filename: r_filename,
            master_file: file,
        }))
    } else {
        std::ptr::null_mut()
    }
}

#[no_mangle]
pub extern "C" fn h5read_free(obj: *mut H5File) {
    let obj = unsafe {
        Box::from_raw(obj);
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
